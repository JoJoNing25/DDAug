import misc
import torch
from torch import nn
import torch.nn.functional as F
import util
from kornia.augmentation import ColorJitter, RandomHorizontalFlip, RandomGrayscale, RandomResizedCrop, RandomGaussianBlur
from kornia.augmentation.container import ImageSequential
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def lid_mle(data, reference, k=20, get_idx=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.e-8), dim=1)
    if get_idx:
        return idx, lids
    return lids


class KorniaSimCLRAugmentation(nn.Module):
    def __init__(self, input_size = (432, 784),
                       min_scale = (0.08, 1.0),
                       cj_strength=1.0, 
                       cj_prob: float = 0.8,
                       random_gray_scale: float = 0.2,
                       hf_prob: float = 0.5,
                       gaussian_blur_prob: float = 0.5,
                       **kwargs):
        super(KorniaSimCLRAugmentation, self).__init__()
        self.aug = ImageSequential(
            RandomResizedCrop(input_size, scale=min_scale),
            RandomHorizontalFlip(p=hf_prob),
            ColorJitter(brightness=cj_strength * 0.8, 
                        contrast=cj_strength * 0.8, 
                        saturation=cj_strength * 0.8, 
                        hue=cj_strength * 0.2,
                        p=cj_prob),
            RandomGrayscale(p=random_gray_scale),
            RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=gaussian_blur_prob),
        )
    
    def forward(self, x):
        x = self.aug(x)
        return x

class KorniaSimCLRCIFARAugmentation(nn.Module):
    def __init__(self, input_size = (32, 32),
                       min_scale = (0.08, 1.0),
                       cj_strength=0.5, 
                       cj_prob: float = 0.8,
                       random_gray_scale: float = 0.2,
                       hf_prob: float = 0.5,
                       **kwargs):
        super(KorniaSimCLRCIFARAugmentation, self).__init__()
        self.aug = ImageSequential(
            RandomResizedCrop(input_size, scale=min_scale),
            RandomHorizontalFlip(p=hf_prob),
            ColorJitter(brightness=cj_strength * 0.8, 
                        contrast=cj_strength * 0.8, 
                        saturation=cj_strength * 0.8, 
                        hue=cj_strength * 0.2,
                        p=cj_prob),
            RandomGrayscale(p=random_gray_scale),
        )
    
    def forward(self, x):
        x = self.aug(x)
        return x
    

class KorniaBaseAugmentation(nn.Module):
    def __init__(self, input_size = (432, 784),
                       min_scale = (0.08, 1.0),
                       hf_prob: float = 0.5,
                       **kwargs):
        super(KorniaBaseAugmentation, self).__init__()
        self.aug = ImageSequential(
            RandomResizedCrop(input_size, scale=min_scale),
            RandomHorizontalFlip(p=hf_prob),
        )
    
    def forward(self, x):
        x = self.aug(x)
        return x


class NTXentLossKornia(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False,
                 augmentations='SimCLR', **kwargs):
        super(NTXentLossKornia, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))
        if augmentations == 'SimCLR':
            self.augmentation = KorniaSimCLRAugmentation(**kwargs)
        elif augmentations == 'SimCLRCIFAR':
            self.augmentation = KorniaSimCLRCIFARAugmentation(**kwargs)
        elif augmentations == 'Base':
            self.augmentation = KorniaBaseAugmentation(**kwargs)
            
    def track_lid(self, f_0, f_1):
        # Track LID
        with torch.no_grad():
            f = torch.cat([f_0, f_1], dim=0).detach()
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.full_gather(f), dim=0)
            else:
                full_rank_f = f

            lids_k32 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=32)
            lids_k512 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=512)
        return lids_k32, lids_k512
    

    def forward(self, model, images):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=False):
            x0, x1 = self.augmentation(images), self.augmentation(images)

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
        
        loss = self.cross_entropy(logits, labels)
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


