import torch
import lid
import util
import misc
import time
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings("ignore")


class mAP_Accumulator(object):
    def __init__(self, num_class=6):
        self.num_class = num_class
        self.reset()

    def update(self, tragets, predictions):
        self.predictions = np.append(self.predictions, predictions, axis=0)
        self.tragets = np.append(self.tragets, tragets, axis=0)

    def reset(self):
        self.predictions = np.empty(shape = [0,self.num_class], dtype=np.float64)
        self.tragets = np.empty(shape = [0,self.num_class], dtype=np.int32)

    def compute(self):
        computed_ap = average_precision_score(self.tragets, self.predictions, average=None)
        actual_ap   = np.mean([x for x in computed_ap if x==x])
        return actual_ap
    

@torch.no_grad()
def evaluate(model, loader, scaler, exp, args):
    model.eval()
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if exp.config.metric == 'binary_multi_cls':
        mAP_calculator = mAP_Accumulator(num_class=exp.config.num_classes)

    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        logits = model(images)
        
        acc5 = None
        if exp.config.metric == 'binary_multi_cls':
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = loss.mean().item()
            mAP_calculator.update(labels.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
            acc = mAP_calculator.compute()
        else:
            loss = F.cross_entropy(logits, labels, reduction='none')
            loss = loss.mean().item()
            # Calculate acc
            acc = util.accuracy(logits, labels, topk=(1,))[0].item()

        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc, n=batch_size)
        if acc5 is not None:
            metric_logger.update(acc5=acc5, n=batch_size)
    metric_logger.synchronize_between_processes()
    loss, acc = metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg
    if acc5 is not None:
        acc5 = metric_logger.meters['acc5'].global_avg
    else:
        acc5 = None
    if exp.config.metric == 'binary_multi_cls':
        acc = mAP_calculator.compute()
    return loss, acc, acc5

def train_epoch(exp, model, optimizer, criterion, scaler, train_loader, global_step, epoch, args, logger):
    epoch_stats = {}
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if exp.config.metric == 'binary_multi_cls':
        mAP_calculator = mAP_Accumulator(num_class=exp.config.num_classes)

    # Training
    for i, data in enumerate(train_loader):
        start = time.time()
        if args.ddp:
            model.module.adjust_train_mode()
        else:
            model.adjust_train_mode()

        if 'lr_schedule_level' in exp.config and exp.config['lr_schedule_level'] == 'epoch':
            util.adjust_learning_rate(optimizer, epoch, exp.config)
        else:
            util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)

        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        model.zero_grad(set_to_none=True)

        # Objective function
        logits = model(images)
        loss = criterion(logits, labels)

        # Optimize
        loss.backward()
        optimizer.step()  
        # Calculate acc
        loss = loss.item()
        acc5 = None
        if exp.config.metric == 'binary_multi_cls':
            mAP_calculator.update(labels.cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy())
            acc = mAP_calculator.compute()
        else:   
            acc = util.accuracy(logits, labels, topk=(1,))[0].item()

        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc, n=batch_size)
        if acc5 is not None:
            metric_logger.update(acc5=acc5, n=batch_size)
        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            loss = misc.all_reduce_mean(loss)
            acc = misc.all_reduce_mean(acc)
            metric_logger.synchronize_between_processes()
            if exp.config.metric == 'binary_multi_cls':
                payload = {
                    "mAP": acc,
                    "mAP_avg": metric_logger.meters['acc'].avg,
                }
            else: 
                payload = {
                    "acc": acc,
                    "acc_avg": metric_logger.meters['acc'].avg,
                }
            payload['loss'] = loss
            payload['loss_avg'] = metric_logger.meters['loss'].avg
            payload['lr'] = optimizer.param_groups[0]['lr']
            
            display = util.log_display(epoch=epoch,
                                       global_step=global_step,
                                       time_elapse=time_used,
                                       **payload)
            if misc.get_rank() == 0:
                logger.info(display)
        # Update Global Step
        global_step += 1
    metric_logger.synchronize_between_processes()
    epoch_stats['epoch'] = epoch
    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].global_avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].global_avg
    return epoch_stats
