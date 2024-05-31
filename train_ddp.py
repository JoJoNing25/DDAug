import argparse
import mlconfig
import torch
import random
import numpy as np
import network
import datasets
import losses
import time
import util
import sys
import os
import misc
from exp_mgmt import ExperimentManager
from collections import OrderedDict

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='JomiSegmentation')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='deeplabv3_rn101', type=str,
                    help='Name of the experiment, config yaml filename')
parser.add_argument('--exp_path', default='experiments',
                    type=str, help='Path to the experiment folder')
parser.add_argument('--exp_config', default='configs/lc_supervised',
                    type=str, help='Path to the config file folder')
parser.add_argument('--load_model', action='store_true', default=False,
                    help='Load model weights from last epoch checkpoint')
parser.add_argument('--load_best_model', action='store_true', default=False,
                    help='Load model weights from best epoch checkpoint')
parser.add_argument('--ddp', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')


def save_model(prefix=''):
    # Save model
    if misc.get_rank() == 0:
        exp.save_state(model, '{}model_state_dict'.format(prefix))
        exp.save_state(optimizer, '{}optimizer_state_dict'.format(prefix))

@torch.no_grad()
def evaluate(loader, epoch=0, key='val'):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="")
    hist_total = None
    for i, val_data in enumerate(loader):
        # Prepare batch data
        if len(val_data) == 2:
            images, labels = val_data
        elif len(val_data) == 4:
            images, labels, image_name, scale_float = val_data
        images = images.to(device)
        labels = labels.to(device)

        output_dict = model(images)
        if type(output_dict) is dict:
            out = output_dict['pred']
        else:
            out, _ = output_dict

        if criterion.weight is not None:
            criterion.weight = criterion.weight.to(device)

        loss = criterion(out, labels)

        # Calculate metrics
        loss = loss.item()
        output_data = torch.nn.functional.softmax(out, dim=1).cpu().data
        max_probs, predictions = output_data.max(1)
        hist = util.fast_hist(predictions.cpu().numpy().flatten(),
                              labels.cpu().numpy().flatten(),
                              exp.config.num_classes)
        iu, acc, acc_cls, f1 = util.calculate_iou(hist)
        miou = np.nanmean(iu)
        mf1 = np.nanmean(f1)

        # Update Meters
        batch_size = images.size(0) * images.size(2) * images.size(3)
        metric_logger.update(loss=loss, n=batch_size)
        metric_logger.update(acc=acc, n=batch_size)
        metric_logger.update(acc_cls=acc_cls, n=batch_size)
        metric_logger.update(miou=miou, n=batch_size)
        metric_logger.update(mf1=mf1, n=batch_size)
        if hist_total is None:
            hist_total = hist
        else:
            hist_total += hist

    hist_total = misc.all_reduce_sum(hist_total).detach().cpu().numpy()
    metric_logger.synchronize_between_processes()
    payload = None

    if misc.get_rank() == 0:
        iu, acc, acc_cls, f1 = util.calculate_iou(hist_total)
        miou_all = np.nanmean(iu)
        id2cat = data.train_set.id_to_cls_name if hasattr(data.train_set, 'id_to_cls_name') else None
        table_results = util.format_evaluate_results(hist_total, iu, epoch=epoch, id2cat=id2cat)

        payload = {
            key+'_statistic': table_results,
            key+'_loss': metric_logger.meters['loss'].avg,
            key+'_acc': acc,
            key+'_acc_cls': acc_cls,
            key+'_miou': miou_all,
            key+'_mf1': f1,
        }
        # Write to Tensorboard
        exp.tb_logger.add_scalar(key+'/loss', metric_logger.meters['loss'].avg, epoch)
        exp.tb_logger.add_scalar(key+'/acc', metric_logger.meters['acc'].avg, epoch)
        exp.tb_logger.add_scalar(key+'/acc_cls', metric_logger.meters['acc_cls'].avg, epoch)
        exp.tb_logger.add_scalar(key+'/mean_iu', miou_all, epoch)
        exp.tb_logger.add_scalar(key+'/mean_f1', f1, epoch)
        # exp.tb_logger.add_scalar(key+'/statistic_2', metric_logger.meters['mf1'].avg, epoch)

    return payload


def train(epoch):
    global global_step
    # Set training metric meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.ddp:
        torch.distributed.barrier()

    # Training
    model.train()
    if args.ddp:
        train_loader.sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust Learning Rate
        util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        # Prepare batch data
        if len(data) == 2:
            images, labels = data
        elif len(data) == 4:
            images, labels, image_name, scale_float = data

        images = images.to(device)
        labels = labels.to(device)

        # Training
        batch_size = images.size(0) * images.size(2) * images.size(3)

        model.zero_grad()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(images)
            if criterion.weight is not None:
                criterion.weight = criterion.weight.to(device)

            loss = criterion(out, labels)

        # Optimize
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
        
        torch.cuda.synchronize()

        # Calculate metrics
        loss = loss.item()
        output_data = torch.nn.functional.softmax(out.detach(), dim=1).cpu().data
        max_probs, predictions = output_data.max(1)
        hist = util.fast_hist(predictions.cpu().numpy().flatten(),
                              labels.cpu().numpy().flatten(),
                              exp.config.num_classes)
        iu, acc, acc_cls, f1 = util.calculate_iou(hist)
        miou = np.nanmean(iu)
        mf1 = np.nanmean(f1)

        # Update Meters
        metric_logger.update(loss=loss, n=batch_size)
        metric_logger.update(acc=acc, n=batch_size)
        metric_logger.update(acc_cls=acc_cls, n=batch_size)
        metric_logger.update(miou=miou, n=batch_size)
        metric_logger.update(mf1=mf1, n=batch_size)
        # Write to Tensorboard
        if misc.get_rank() == 0:
            exp.tb_logger.add_scalar('train/loss', loss, global_step)
            exp.tb_logger.add_scalar('train/acc', acc, global_step)
            exp.tb_logger.add_scalar('train/acc_cls', acc_cls, global_step)
            exp.tb_logger.add_scalar('train/mean_iu', miou, global_step)
            exp.tb_logger.add_scalar('train/mean_f1', mf1, global_step)

        # track LR
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            loss = misc.all_reduce_mean(loss)
            metric_logger.synchronize_between_processes()
            if misc.get_rank() == 0:
                payload = {
                    "loss": loss,
                    "loss_avg": metric_logger.meters['loss'].avg,
                    "miou_avg": metric_logger.meters['miou'].avg,
                    "acc_avg": metric_logger.meters['acc'].avg,
                    "acc_cls_avg": metric_logger.meters['acc_cls'].avg,
                    "mf1_avg": metric_logger.meters['mf1'].avg,
                    "min_lr": min_lr,
                    'max_lr': max_lr,
                }
                display = util.log_display(epoch=epoch,
                                           global_step=global_step,
                                           time_elapse=time_used,
                                           **payload)
                logger.info(display)

        # Update Global Step
        global_step += 1

    metric_logger.synchronize_between_processes()
    payload = {
        "loss_avg": metric_logger.meters['loss'].avg,
        "miou_avg": metric_logger.meters['miou'].avg,
        "acc_avg": metric_logger.meters['acc'].avg,
        "acc_cls_avg": metric_logger.meters['acc_cls'].avg,
        "mf1_avg": metric_logger.meters['mf1'].avg,
        "min_lr": min_lr,
        'max_lr': max_lr,
    }
    return payload


def main():
    # Set Global Vars
    global model, optimizer, criterion, model_without_ddp, scaler
    global train_loader, test_loader, val_loader, data
    global logger, start_epoch, global_step, best_metric

    # Set up Experiments
    logger = exp.logger
    config = exp.config

    # Prepare Data
    data = config.dataset(exp, seed=args.seed)
    if misc.get_rank() == 0:
        logger.info('Train size %d' % len(data.train_set))
        logger.info('Val size %d' % len(data.val_set))
        logger.info('Test size %d' % len(data.test_set))

    if args.ddp:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            data.train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(data.test_set) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                data.val_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                data.test_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(data.val_set)
            sampler_test = torch.utils.data.SequentialSampler(data.test_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.val_set)
        sampler_test = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(train_sampler=sampler_train, val_sampler=sampler_val, test_sampler=sampler_test)
    train_loader, val_loader, test_loader = loader

    # Prepare Model and loss func
    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None
    model = config.model().to(device)
    optimizer = config.optimizer(model.parameters())
    criterion = config.criterion()
    print(model)
    from thop import profile
    profile_inputs = (torch.rand(16,3,304,304).cuda(),)
    flops, params = profile(model, inputs=profile_inputs, verbose=False)
    print(flops / 1e6, params / 1e6)
    print('number of weights:', sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e-6)

    start_epoch = 0
    global_step = 0
    best_epoch = 0
    best_val_metric = 0
    best_test_epoch = 0
    best_test_metric = 0

    # Load pretrain weights
    if 'pretrain_weight' in exp.config:
        if exp.config.pretrain_weight == 'pretrain':
            path = exp.exp_path.replace(exp.exp_name, 'pretrain')
            path = os.path.join(path, 'checkpoints/model_state_dict.pt')
        else:
            path = exp.config.pretrain_weight
        model = model.to('cpu')
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                remove_l = len('module.')
                name = k[remove_l:]  # remove encoder.
            else:
                name = k
            if name.startswith('fc') or name.startswith('final') or name.startswith('final.6') or name.startswith('decoder_pred'):
                # Ignore FC
                continue
            if 'pos_embed' in name:
                # Ignore pos_embed
                continue
            new_state_dict[name] = v
        msg = model.load_state_dict(new_state_dict, strict=False)
        if misc.get_rank() == 0:
            logger.info(msg)
            print(msg)
        del new_state_dict, state_dict
        # Init Finetune weights
        model.finetune()

    # Adjust Layer Decays
    model = model.to(device)
    # param_groups = util.param_groups_lrd(model, weight_decay=exp.config.weight_decay,
    #                                      layer_decay=exp.config.layer_decay)
    optimizer = config.optimizer(model.parameters())

    # Resume: Load models
    if args.load_model:
        exp_stats = exp.load_epoch_stats()
        start_epoch = exp_stats['epoch'] + 1
        global_step = exp_stats['global_step'] + 1
        model = exp.load_state(model, 'model_state_dict')
        optimizer = exp.load_state(optimizer, 'optimizer_state_dict')
        best_metric = exp_stats['best_metric']
        best_epoch = exp_stats['best_epoch']

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, broadcast_buffers=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=True)
        model_without_ddp = model.module

    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        stats = train(epoch)

        # Epoch Val
        is_best = False
        is_best_test = False
        val_results = {}
        test_results = {}
        if ('eval_freq' in exp.config and epoch != 0 and epoch % exp.config.eval_freq == 0) or \
                'eval_freq' not in exp.config or epoch == exp.config.epochs - 1:
            if misc.get_rank() == 0:
                logger.info("="*20 + "Validation Epoch %d" % (epoch) + "="*20)

            val_results = evaluate(val_loader, epoch, key='val')
            if misc.get_rank() == 0:
                logger.info('\033[33m\n %s \033[0m' % (val_results['val_statistic']))
                logger.info('\033[33mVal Loss: %.4f \033[0m' % (val_results['val_loss']))
                logger.info('\033[33mVal Acc: %.4f \033[0m' % (val_results['val_acc']))
                logger.info('\033[33mVal Acc CLS: %.4f \033[0m' % (val_results['val_acc_cls']))
                # logger.info('\033[33mVal MIoU: %.4f \033[0m' % (val_results['val_miou']))
                # logger.info('\033[33mVal MF1: %.4f \033[0m' % (val_results['val_mf1']))
                logger.info('\033[33mVal MIoU v2: %.4f \033[0m' % (val_results['val_miou']))
                logger.info('\033[33mVal MF1 v2: %.4f \033[0m' % (val_results['val_mf1']))
                logger.info('\033[33mPrev Best MIoU: %.4f \033[0m' % (best_val_metric))
                is_best = val_results['val_miou'] > best_val_metric
                best_val_metric = max(best_val_metric, val_results['val_miou'])
                if is_best:
                    best_epoch = epoch
                stats['best_val_miou'] = best_val_metric
                stats['best_val_mf1'] = val_results['val_mf1']
                stats['best_val_acc'] = val_results['val_acc']
                stats['best_val_acc_cls'] = val_results['val_acc_cls']

            # Epoch Test
            if misc.get_rank() == 0:
                logger.info("="*20 + "Test Epoch %d" % (epoch) + "="*20)

            test_results = evaluate(test_loader, epoch, key='test')

            if misc.get_rank() == 0:
                logger.info('\033[33m\n %s \033[0m' % (test_results['test_statistic']))
                logger.info('\033[33mTest Loss: %.4f \033[0m' % (test_results['test_loss']))
                logger.info('\033[33mTest Acc: %.4f \033[0m' % (test_results['test_acc']))
                logger.info('\033[33mTest Acc CLS: %.4f \033[0m' % (test_results['test_acc_cls']))
                # logger.info('\033[33mTest MIoU: %.4f\033[0m' % (test_results['test_miou']))
                # logger.info('\033[33mTest MF1: %.4f\033[0m' % (test_results['test_mf1']))
                logger.info('\033[33mTest MIoU v2: %.4f\033[0m' % (test_results['test_miou']))
                logger.info('\033[33mTest MF1 v2: %.4f\033[0m' % (test_results['test_mf1']))
                logger.info('\033[33mPrev Best MIoU: %.4f \033[0m' % (best_test_metric))
                is_best_test = test_results['test_miou'] > best_test_metric
                best_test_metric = max(best_test_metric, test_results['test_miou'])
                if is_best_test:
                    best_test_epoch = epoch
                stats['best_test_miou'] = best_test_metric
                stats['best_test_mf1'] = test_results['test_mf1']
                stats['best_test_acc'] = test_results['test_acc']
                stats['best_test_acc_cls'] = test_results['test_acc_cls']

        # Save Model and exp stats
        if misc.get_rank() == 0:
            save_model()
            if is_best:
                save_model(prefix='best_val_')
            if is_best_test:
                save_model(prefix='best_test_')
            stats['epoch'] = epoch
            stats['global_step'] = global_step
            stats['best_epoch'] = best_epoch
            stats['best_test_epoch'] = best_test_epoch
            stats.update(val_results)
            stats.update(test_results)
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
    return


if __name__ == '__main__':
    global exp
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
        logger.info("World Size: %d" % (misc.get_world_size()))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    misc.destroy_process_group()
