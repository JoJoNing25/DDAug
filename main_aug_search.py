import argparse
import torch
import mlconfig
import network
import datasets
import losses
import util
import misc
import os
import sys
import numpy as np
import time
import math
from lid import gmean
from exp_mgmt import ExperimentManager
from engine_aug_search import train_epoch
from collections import OrderedDict
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='SSL-LID')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')


def save_model(model, optimizer, epoch=None):
    if args.ddp:
        model = model.module
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if epoch is not None:
        exp.save_state(model, 'model_state_dict_epoch{:d}'.format(epoch))


def main():
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
                data.val_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                data.test_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(data.val_set)
            sampler_test = torch.utils.data.SequentialSampler(data.test_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.val_set)
        sampler_test = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(train_sampler=sampler_train, val_sampler=sampler_val,
                             test_sampler=sampler_test)
    train_loader, val_loader, test_loader = loader

    # Prepare Model
    model = config.model()
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

    model = model.to(device)
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Search Policy
    policy = config.policy().to(device).to(device)
    optimizer = config.optimizer(policy.parameters())
    
    if misc.get_rank() == 0:
        print(policy)
    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        policy = torch.nn.parallel.DistributedDataParallel(policy)
        
    start_epoch = 0
    global_step = 0

    # Train Loops
    if args.ddp:
        policy_sample = policy.module._sample()
    else:
        policy_sample = policy._sample()
    if misc.get_rank() == 0:
        logger.info('\033[33m initial policy: '+str(policy_sample)+'\033[0m')
    for epoch in range(start_epoch, exp.config.epochs):
        start_time = time.time()
        stats = {}
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        stats = train_epoch(exp, model, policy, optimizer, scaler, train_loader, global_step, epoch, args, logger)            
        global_step = stats['global_step']        
        # Save Model
        if misc.get_rank() == 0:
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
            save_model(policy, optimizer)
            
        end_time = time.time()
        cost_per_epoch = (end_time - start_time) / 60
        esitmited_finish_cost = (end_time - start_time) / 3600 * (exp.config.epochs - epoch - 1)
        if misc.get_rank() == 0:
            payload = "Running Cost %.2f mins/epoch, finish in %.2f hours (esimitated)" % (cost_per_epoch, esitmited_finish_cost)
            logger.info('\033[33m'+payload+'\033[0m')
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
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    misc.destroy_process_group()