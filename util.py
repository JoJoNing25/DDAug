import logging
import os
import numpy as np
import torch
import json
import math
from torch import inf
from decimal import Decimal
from tabulate import tabulate
from scipy.spatial.distance import cdist
from torcheval.metrics import MulticlassConfusionMatrix
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def round_num(n, decimal=2):
    if n == 0:
        return 0
    n = Decimal(n)
    return round(n.normalize(), decimal)


def numerize(n, decimal=2):
    # https://github.com/davidsa03/numerize/blob/master/numerize/numerize.py
    #60 sufixes
    sufixes = [ "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No", 
                "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V", 
                "Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
                "Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
                "Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
                "Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"] 
    
    sci_expr = [1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27, 
                1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57, 
                1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87, 
                1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117, 
                1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147, 
                1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]
    minus_buff = n
    n = abs(n)
    if n == 0:
        return str(n)
    for x in range(len(sci_expr)):
        try:
            if n >= sci_expr[x] and n < sci_expr[x+1]:
                sufix = sufixes[x]
                if n >= 1e3:
                    num = str(round_num(n/sci_expr[x], decimal))
                else:
                    num = str(n)
                return num + sufix if minus_buff > 0 else "-" + num + sufix
        except IndexError:
            print("You've reached the end")

def update_momentum(model, model_ema, m):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`
    Momentum encoders are a crucial component fo models such as MoCo or BYOL.
    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def exclude_from_wd_and_adaptation(name, bn=True):
        if 'bn' in name and bn:
            return True
        if 'bias' in name:
            return True


def get_lars_params(model, weight_decay, bn=True):
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name, bn)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name, bn)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


def param_layers_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    param_group_names = {}
    param_groups = {}
    num_layers = len(list(model.named_parameters())) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    for layer_id, (name, p) in enumerate(list(model.named_parameters())):
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        this_decay = weight_decay

        if name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[name]["params"].append(name)
        param_groups[name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'predictor' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs.warmup_epochs:
        lr = configs.lr * epoch / configs.warmup_epochs
    elif 'lr_schedule' in configs:
        if configs.lr_schedule == 'milestone':
            milestones = [int(s*configs.epochs)for s in [0.65, 0.85, 0.95]]
            if epoch < milestones[0]:
                lr = configs.lr
            elif epoch >= milestones[0] and epoch < milestones[1]:
                lr = configs.lr * 0.1
            else:
                lr = configs.lr * 0.01  
        elif configs.lr_schedule == 'linear':
            # lr = np.maximum(configs.lr * np.minimum(configs.epochs / epoch + 1., 1.), 0.)
            lr = configs.lr * (1 - (epoch - configs.warmup_epochs) / (configs.epochs - configs.warmup_epochs))
        elif configs.lr_schedule == 'cosine':
            min_lr = configs.lr * configs.min_lr
            lr = min_lr + (configs.lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - configs.warmup_epochs) / (configs.epochs - configs.warmup_epochs)))
    else:
        min_lr = configs.lr * configs.min_lr
        lr = min_lr + (configs.lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - configs.warmup_epochs) / (configs.epochs - configs.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        elif 'fix_lr' in param_group and param_group['fix_lr']:
            param_group["lr"] = configs.lr
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_with_params(optimizer, epoch, min_lr, lr, warmup, epochs, lr_schedule=None):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup:
        lr = lr * epoch / warmup
    elif lr_schedule:
        if lr_schedule == 'milestone':
            milestones = [int(s*epochs)for s in [0.65, 0.85, 0.95]]
            if epoch < milestones[0]:
                lr = lr
            elif epoch >= milestones[0] and epoch < milestones[1]:
                lr = lr * 0.1
            else:
                lr = lr * 0.01  
        elif lr_schedule == 'linear':
            lr = lr * (1 - (epoch - warmup) / (epochs - warmup))
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        elif 'fix_lr' in param_group and param_group['fix_lr']:
            print(param_group, lr)
            param_group["lr"] = lr
        else:
            param_group["lr"] = lr
    return lr


def adjust_lid_targets(criterion, epoch, epochs):
    criterion.lid_target = criterion.lid_min + (criterion.lid_max - criterion.lid_min) * 0.5 * \
        (1. + math.cos(math.pi * epoch / epochs))
    return criterion.lid_target


def setup_logger(name, log_file, ddp=False, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    if not ddp:
        logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              ' global_step=' + str(numerize(global_step, 1))
    for key, value in kwargs.items():
        if type(value) == str:
            display = ' ' + key + '=' + value
        else:
            if 'loss' in key or 'acc' in key or 'lr' in key: 
                display += ' ' + str(key) + '=%.4f' % value
            else:
                display += ' ' + str(key) + '=%.2f' % value
    display += ' time=%.1fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res


def multi_label_acc(output, target, threshold=0.5):
    pred = torch.sigmoid(output)
    pred = (pred > threshold)
    correct = (pred.long()==target.long()).float()
    cls_acc = correct.mean(dim=1)
    acc = cls_acc.mean()
    return acc

def mean_cls_acc(output, target):
    acc = [0 for c in range(output.shape[1])]
    _, preds = torch.max(output.data, 1)
    for c in range(output.shape[1]):
        acc[c] = ((preds == target) * (target == c)).sum().float() / max((target == c).sum(), 1)
    return sum(acc) / len(acc)


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    if model.__class__.__name__ != 'ViTDeeplabV3':
        layer_ids = model._get_layer_ids()
        layer_scales = list(layer_decay ** (len(layer_ids) - i - 1) for i in range(len(layer_ids)+1))
        param_groups = {}
        param_group_names = {}
        used = []
        for n, p in model.named_parameters():
            for i, layer_name in enumerate(layer_ids):
                if layer_name in n and n not in used and not n[len(layer_name)].isdigit():
                    this_scale = layer_scales[i]
                    if layer_name not in param_groups:
                        param_groups[layer_name] = {
                            "lr_scale": this_scale,
                            "weight_decay": weight_decay,
                            "params": [p],
                        }
                        param_group_names[layer_name] = {
                            "lr_scale": this_scale,
                            "weight_decay": weight_decay,
                            "params": [n],
                        }
                    else:
                        param_groups[layer_name]['params'].append(p)
                        param_group_names[layer_name]['params'].append(n)
                    used.append(n)
    else:
        param_groups = {}
        param_group_names = {}
        num_layers = len(model.blocks) + 1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # no decay: all 1D parameters and model specific ones
            if p.ndim == 1 or n in no_weight_decay_list:
                g_decay = "no_decay"
                this_decay = 0.
            else:
                g_decay = "decay"
                this_decay = weight_decay

            layer_id = get_layer_id_for_vit(n, num_layers)
            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                 }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                 }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def format_evaluate_results_v2(hist, epoch=0, iou_per_scale=[1], eps=1e-8, id2cat=None):
    """
    If single scale:
       just print results for default scale
    else
       print all scale results

    Inputs:
    hist = histogram for default scale
    iu = IOU for default scale
    iou_per_scale = iou for all scales
    row=gt
    column=pd
    """
    iu, _, _, _ = calculate_iou(hist)
    if id2cat is not None:
        pass
    else:
        id2cat = {i: i for i in range(len(iu))}

    pixels_per_class = hist.sum(axis=0)
    predicts_per_class = hist.sum(axis=1)

    iu_FP = hist.sum(axis=0) - np.diag(hist)
    iu_FN = hist.sum(axis=1) - np.diag(hist)
    iu_TP = np.diag(hist)

    header = ['Id', 'label']
    header.extend(['iU_{}'.format(scale) for scale in iou_per_scale])
    header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])

    tabulate_data = []

    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)
        for scale in iou_per_scale:
            class_data.append(iu[class_id] * 100)
        total_pixels = hist.sum()
        # Jojo good, Jojo beautiful
        class_data.append(100 * iu_TP[class_id] / pixels_per_class[class_id])
        class_data.append(100 * iu_FP[class_id] / pixels_per_class[class_id])
        class_data.append(100 * iu_FN[class_id] / predicts_per_class[class_id])
        precision = iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id] + eps)
        class_data.append(precision)
        recall = iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id] + eps)
        class_data.append(recall)
        class_data.append(2*precision*recall/(precision+recall))
        tabulate_data.append(class_data)

    return str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))


def format_evaluate_results(hist, iu, epoch=0, iou_per_scale=[1], eps=1e-8, id2cat=None):
    """
    If single scale:
       just print results for default scale
    else
       print all scale results

    Inputs:
    hist = histogram for default scale
    iu = IOU for default scale
    iou_per_scale = iou for all scales
    row=gt
    column=pd
    """
    if id2cat is not None:
        pass
    else:
        id2cat = {i: i for i in range(len(iu))}

    pixels_per_class = hist.sum(axis=0)
    predicts_per_class = hist.sum(axis=1)

    iu_FP = hist.sum(axis=0) - np.diag(hist)
    iu_FN = hist.sum(axis=1) - np.diag(hist)
    iu_TP = np.diag(hist)

    header = ['Id', 'label']
    header.extend(['iU_{}'.format(scale) for scale in iou_per_scale])
    header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])

    tabulate_data = []

    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)
        for scale in iou_per_scale:
            class_data.append(iu[class_id] * 100)
        total_pixels = hist.sum()
        class_data.append(100 * iu_TP[class_id] / pixels_per_class[class_id])
        class_data.append(100 * iu_FP[class_id] / pixels_per_class[class_id])
        class_data.append(100 * iu_FN[class_id] / predicts_per_class[class_id])
        precision = iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id] + eps)
        class_data.append(precision)
        recall = iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id] + eps)
        class_data.append(recall)
        class_data.append(2*precision*recall/(precision+recall))
        tabulate_data.append(class_data)

    return str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist


def calculate_iou(hist_data, eps=1e-8):
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - \
        np.diag(hist_data)
    iu = np.diag(hist_data) / divisor
    iu_FP = hist_data.sum(axis=0) - np.diag(hist_data)
    iu_FN = hist_data.sum(axis=1) - np.diag(hist_data)
    iu_TP = np.diag(hist_data)
    precision = iu_TP.sum() / (iu_TP + iu_FP + eps).sum()
    recall = iu_TP.sum() / (iu_TP + iu_FN + eps).sum()
    f1 = 2*precision*recall/(precision+recall)
    return iu, acc, acc_cls, f1


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
                    - np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
                    - np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
