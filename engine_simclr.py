import torch
import lid
import util
import misc
import time
import math
from lid import gmean
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


@torch.no_grad()
def evaluate(model, test_loader, args, configs):
    model.eval()
    # extract features
    lids_k32 = []
    lids_k512 = []
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        fe = out[0]
        cls = out[2]
        if args.ddp:
            full_rank_fe = torch.cat(misc.gather(fe), dim=0)
        else:
            full_rank_fe = fe
        lids_k32.append(lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=32))
        lids_k512.append(lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=512))

    lids_k32 = torch.cat(lids_k32, dim=0)
    lids_k512 = torch.cat(lids_k512, dim=0)
    if args.ddp:
        lids_k32 = torch.cat(misc.gather(lids_k32.to(device)), dim=0)
        lids_k512 = torch.cat(misc.gather(lids_k512.to(device)), dim=0)
    
    return lids_k32.detach().cpu(), lids_k512.detach().cpu()


def train_epoch(exp, model, optimizer, criterion, scaler, train_loader, global_step, epoch, args, logger):
    epoch_stats = {}
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Training
    model.train()
    for param in model.parameters():
        param.grad = None
    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust LR
        util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        # Train step
        images = data[0]
        model.train()
        model.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = criterion(model, images)
            loss = results['loss']
            logits = results['logits']
            labels = results['labels']
        # Optimize
        loss = results['loss']
        if torch.isnan(loss):
            if misc.get_rank() == 0:
                logger.info('Skip this batch, loss is nan!')
            raise('loss is nan!')
        if scaler is not None:
            scaler.scale(loss).backward()
            if hasattr(exp.config, 'grad_clip'):
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                if hasattr(exp.config, 'grad_clip'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if hasattr(exp.config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            optimizer.step()
        loss = loss.item()
        # Calculate acc
        acc, _ = util.accuracy(logits, labels, topk=(1, 5))
        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc.item(), n=batch_size)
        metric_logger.update(lid32_avg=results['lids32'].mean().item())
        metric_logger.update(lid32_var=results['lids32'].var().item())
        metric_logger.update(lid512_avg=results['lids512'].mean().item())
        metric_logger.update(lid512_var=results['lids512'].var().item())
        metric_logger.update(lid32_gavg=gmean(results['lids32']).item())
        metric_logger.update(lid512_gavg=gmean(results['lids512']).item())
        metric_logger.update(main_loss=results['main_loss'])
        if 'reg_loss' in results:
            metric_logger.update(reg_loss=results['reg_loss'])
        # Log results
        end = time.time()
        time_used = end - start
        # track LR
        lr = optimizer.param_groups[0]['lr']
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
                "acc_avg": metric_logger.meters['acc'].avg,
                "lid32_gavg": metric_logger.meters['lid32_gavg'].avg,
                "lid512_gavg": metric_logger.meters['lid512_gavg'].avg,
                "loss_avg": metric_logger.meters['loss'].avg,
                "main_loss": metric_logger.meters['main_loss'].avg,
            }
            if 'reg_loss' in results:
                payload['reg_loss'] = metric_logger.meters['reg_loss'].avg
            if misc.get_rank() == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=global_step,
                                           time_elapse=time_used,
                                           **payload)
                logger.info(display)
        # Update Global Step
        global_step += 1

    metric_logger.synchronize_between_processes()
    epoch_stats['epoch'] = epoch
    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].global_avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].global_avg
    epoch_stats['train_lid32_avg'] = metric_logger.meters['lid32_avg'].global_avg
    epoch_stats['train_lid32_var'] = metric_logger.meters['lid32_var'].global_avg
    epoch_stats['train_lid512_avg'] = metric_logger.meters['lid512_avg'].global_avg
    epoch_stats['train_lid512_var'] = metric_logger.meters['lid512_var'].global_avg
    epoch_stats['train_lid32_gavg'] = metric_logger.meters['lid32_gavg'].global_avg
    epoch_stats['train_lid512_gavg'] = metric_logger.meters['lid512_gavg'].global_avg
    epoch_stats['main_loss'] = metric_logger.meters['main_loss'].global_avg
    if 'reg_loss' in results:
        epoch_stats['reg_loss'] = metric_logger.meters['reg_loss'].global_avg
    return epoch_stats
