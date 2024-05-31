import torch
import lid
import util
import misc
import time
import random
import torch.nn.functional as F
from lid import gmean
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
from kornia.augmentation.container import ImageSequential
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train_epoch(exp, model, policy, optimizer, scaler, train_loader, global_step, epoch, args, logger):
    epoch_stats = {}
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Training
    policy.train()
    model.eval()
    
    if hasattr(exp.config, 'pre_aug_ops_size'):
        pre_aug_ops = ImageSequential(
            RandomResizedCrop(exp.config.pre_aug_ops_size, scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5)
        )
    else:
        pre_aug_ops = ImageSequential(
            RandomResizedCrop((432, 784), scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5)
        )
    
    for i, data in enumerate(train_loader):
        
        start = time.time()
        # Adjust LR
        util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        # Train step
        images = data[0]
        images = pre_aug_ops(images).to(device)
        
        model.eval()
        policy.train()
        policy.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            features, _ = model(policy(images))
            if args.ddp:
                full_rank_features = torch.cat(misc.gather(features), dim=0)
            else:
                full_rank_features = features
            lids = lid.lid_mom_est(features, full_rank_features.detach(), k=exp.config.lid_k)
            loss = - torch.abs(torch.log(lids/1)).mean()

            # Optimize
            if torch.isnan(loss):
                if misc.get_rank() == 0:
                    logger.info('Skip this batch, loss is nan!')
                raise('loss is nan!')
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss = loss.item()
        # Update Meters
        metric_logger.update(loss=loss)
        metric_logger.update(lids_mean=lids.mean().item())
        metric_logger.update(lids_geometric_mean=gmean(lids).item())

        # Log results
        end = time.time()
        time_used = end - start
        # track LR
        lr = optimizer.param_groups[0]['lr']
        
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
                "loss_avg": metric_logger.meters['loss'].avg,
                "lids_mean": metric_logger.meters['lids_mean'].avg,
                "lids_geometric_mean": metric_logger.meters['lids_geometric_mean'].avg,
            }
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
    epoch_stats['loss'] = metric_logger.meters['loss'].global_avg
    epoch_stats['lids_mean'] = metric_logger.meters['lids_mean'].global_avg
    epoch_stats['lids_geometric_mean'] = metric_logger.meters['lids_geometric_mean'].global_avg
    if args.ddp:
        policy_sample = policy.module._sample()
    else:
        policy_sample = policy._sample()
    epoch_stats['policy'] = str(policy_sample)
    if misc.get_rank() == 0:
        logger.info('\033[33m'+epoch_stats['policy']+'\033[0m')

    return epoch_stats


def train_epoch_min_max_loss(exp, model, linear_predic_head, policy, optimizer, scaler, train_loader, global_step, epoch, args, logger):
    epoch_stats = {}
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Training
    policy.train()
    model.eval()

    
    if hasattr(exp.config, 'pre_aug_ops_size'):
        pre_aug_ops = ImageSequential(
            RandomResizedCrop(exp.config.pre_aug_ops_size, scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5)
        )
    else:
        pre_aug_ops = ImageSequential(
            RandomResizedCrop((432, 784), scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5)
        )
    
    for i, data in enumerate(train_loader):
        
        start = time.time()
        # Adjust LR
        util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)

        # Train step
        images = data[0]
        images = pre_aug_ops(images).to(device)
        
        model.eval()
        policy.train()
        policy.zero_grad()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            images_0 = policy(pre_aug_ops(images).to(device))
            features, z_0 = model(images_0)
            with torch.no_grad():
                # Due to gpu memory limitation
                _, z_1 = model(policy(pre_aug_ops(images).to(device)))

            if args.ddp:
                full_rank_features = torch.cat(misc.gather(features), dim=0)
            else:
                full_rank_features = features
            lids = lid.lid_mom_est(features, full_rank_features.detach(), k=exp.config.lid_k)

            # Calculate INFO NCE loss
            batch_size = z_0.shape[0]
            z_0 = F.normalize(z_0, dim=1)
            z_1 = F.normalize(z_1, dim=1)
            
            # user other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if args.ddp and misc.world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(misc.full_gather(z_0), 0)
                out1_large = torch.cat(misc.full_gather(z_1), 0)
                diag_mask = misc.eye_rank(batch_size, device=z_0.device)
            else:
                # single process
                out0_large = z_0
                out1_large = z_1
                diag_mask = torch.eye(batch_size, device=z_0.device, dtype=torch.bool)
            
            # calculate similiarities
            # here n = batch_size and m = batch_size * world_size
            # the resulting vectors have shape (n, m)
            logits_00 = torch.einsum('nc,mc->nm', z_0, out0_large) / 0.1
            logits_01 = torch.einsum('nc,mc->nm', z_0, out1_large) / 0.1
            logits_10 = torch.einsum('nc,mc->nm', z_1, out0_large) / 0.1
            logits_11 = torch.einsum('nc,mc->nm', z_1, out1_large) / 0.1

            # remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # concatenate logits
            # the logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = labels + misc.rank() * batch_size
            labels = labels.repeat(2)
            
            info_nce_loss = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
            
            # Rotation Loss
            angle = random.randint(0, 3)
            images_0 = torch.rot90(images_0, angle, [2, 3])
            images_0 = images_0.to(device, non_blocking=True)
            labels = torch.zeros(images_0.shape[0], device=device).long() + angle

            logits = linear_predic_head(model(images_0)[0].view(batch_size, -1))
            rotation_loss = torch.nn.functional.cross_entropy(logits, labels)

            loss = rotation_loss - info_nce_loss
            loss = loss.mean(dim=0)

            # Optimize
            if torch.isnan(loss):
                if misc.get_rank() == 0:
                    logger.info('Skip this batch, loss is nan!')
                raise('loss is nan!')
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss = loss.item()
        # Update Meters
        metric_logger.update(loss=loss)
        metric_logger.update(lids_mean=lids.mean().item())
        metric_logger.update(lids_geometric_mean=gmean(lids).item())

        # Log results
        end = time.time()
        time_used = end - start
        # track LR
        lr = optimizer.param_groups[0]['lr']
        
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
                "loss_avg": metric_logger.meters['loss'].avg,
                "lids_mean": metric_logger.meters['lids_mean'].avg,
                "lids_geometric_mean": metric_logger.meters['lids_geometric_mean'].avg,
            }
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
    epoch_stats['loss'] = metric_logger.meters['loss'].global_avg
    epoch_stats['lids_mean'] = metric_logger.meters['lids_mean'].global_avg
    epoch_stats['lids_geometric_mean'] = metric_logger.meters['lids_geometric_mean'].global_avg
    if args.ddp:
        policy_sample = policy.module._sample()
    else:
        policy_sample = policy._sample()
    epoch_stats['policy'] = str(policy_sample)
    if misc.get_rank() == 0:
        logger.info('\033[33m'+epoch_stats['policy']+'\033[0m')

    return epoch_stats
