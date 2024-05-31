import misc
import torch
from torch import nn
import torch.nn.functional as F
import util
import lid
from datasets.aug_search_ops import AugPolicy
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
from kornia.augmentation.container import ImageSequential

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NTXentLossLIDAug(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False,
                 aug_layers=5, aug_temperature=1.0, sample_mode='argmax', search_space='S1', augmentations='', **kwargs):
        super(NTXentLossLIDAug, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))
        self.augmentation = AugPolicy(layers=aug_layers, temperature=aug_temperature, sample_mode=sample_mode, search_space=search_space)
        if augmentations == 'None':
            pass
        else:
            augmentation_weights = torch.load(augmentations)
            self.augmentation.load_state_dict(augmentation_weights)
        self.augmentation.eval()
        if 'input_size' in kwargs:
            input_size = kwargs['input_size']
        else:
            input_size = (432, 784)
        self.pre_aug_ops = ImageSequential(
            RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5)
        )
        for param in self.augmentation.parameters():
            param.requires_grad = False
            
    def track_lid(self, f_0, f_1):
        # Track LID
        with torch.no_grad():
            f = torch.cat([f_0, f_1], dim=0).detach()
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.full_gather(f), dim=0)
            else:
                full_rank_f = f

            lids_k32 = lid.lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=32)
            lids_k512 = lid.lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=512)
        return lids_k32, lids_k512
    

    def forward(self, model, images):
        images = images.to(device, non_blocking=True)
        
        with torch.no_grad():
            x0, x1 = self.augmentation(self.pre_aug_ops(images)), self.augmentation(self.pre_aug_ops(images))
            
        f_0, z_0 = model(x0)
        f_1, z_1 = model(x1)
        
        batch_size = z_0.shape[0]
        z_0 = F.normalize(z_0, dim=1)
        z_1 = F.normalize(z_1, dim=1)
        
        # user other samples from batch as negatives
        # and create diagonal mask that only selects similarities between
        # views of the same image
        if self.gather_distributed and misc.world_size() > 1:
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
        logits_00 = torch.einsum('nc,mc->nm', z_0, out0_large) / self.temperature
        logits_01 = torch.einsum('nc,mc->nm', z_0, out1_large) / self.temperature
        logits_10 = torch.einsum('nc,mc->nm', z_1, out0_large) / self.temperature
        logits_11 = torch.einsum('nc,mc->nm', z_1, out1_large) / self.temperature

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
        
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        loss = self.cross_entropy(logits, labels)
        loss = torch.nan_to_num(loss, nan=0.1, posinf=0.1, neginf=0.1)
        loss = loss.mean(dim=0)
        # Track LID
        lids_k32, lids_k512 = self.track_lid(f_0, f_1)
        results = {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "main_loss": loss.item(),
        }
        return results


