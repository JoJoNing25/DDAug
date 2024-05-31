import mlconfig
import torch
from . import deepv3, simclr, lars
mlconfig.register(deepv3.DeepV3Plus)
mlconfig.register(deepv3.DeepV3PlusDAE)
mlconfig.register(simclr.SimCLR)
# optim
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(lars.LARS)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
