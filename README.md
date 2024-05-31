# DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery

Code for MIDL2024 paper ["DDA: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery"](https://arxiv.org/abs/2401.10474)

---
## DDAug pipeline

```python
# Method of Moments estimation of LID
def lid_mom_est(data, reference, k, get_idx=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / (a[:, k] - m)
    if get_idx:
        return idx, lids
    return lids

# features: representations that need LID to be estimated. 
# reference: reference representations, usually, the same batch of representations can be used. 
# k: locality parameter, the neighbourhood size. 
# NOTE: features and reference should be in the same dimension.

lids = lid_mom_est(data=features, reference=full_rank_features.detach(), k=k)
loss = - torch.abs(torch.log(lids/1)).mean() # Eq (5) of the paper. 
        
```

---
## Reproduce results from the paper
We provide configuration files in the *configs* folder. Details of all necessary hyperparameters are also in the Appendix of the paper. 

Pretrained models are available here in this [Google Drive](https://drive.google.com/drive/folders/1-djggRfCqIVTbY-gFkJ7C8fGHQEVFAtN?usp=sharing) folder. 

An example of how to pretrain the base model:
```
srun python3 -u main_simclr_kornia.py --exp_name      $exp_name     \
                                      --exp_path      $exp_path     \
                                      --exp_config    $exp_config   \
                                      --ddp --dist_eval --seed $seed
```

An example of how to run an augmentation search:
```
srun python3 -u main_aug_search.py --exp_name      $exp_name     \
                                   --exp_path      $exp_path     \
                                   --exp_config    $exp_config   \
                                   --ddp --dist_eval --seed $seed        
```

An example of how to run linear probing:
```
srun python3 -u main_linear_prob.py --exp_name      $exp_name     \
                                    --exp_path      $exp_path     \
                                    --exp_config    $exp_config   \
                                    --seed          $seed         \
                                    --ddp        
```

An example of how to run finetuning:
```
srun python3 -u train_ddp.py --exp_name      $exp_name     \
                             --exp_path      $exp_path     \
                             --exp_config    $exp_config   \
                             --seed          $seed         \
                             --ddp     
```



## Citation
If you use this code in your work, please cite the accompanying paper:
```
@inproceedings{
zhou2024dda,
title={{DDA}: Dimensionality Driven Augmentation Search for Contrastive Learning in Laparoscopic Surgery},
author={Yuning Zhou and Henry Badgery and Matthew Read and James Bailey and Catherine Davey},
booktitle={Medical Imaging with Deep Learning},
year={2024}
}

```

## Part of the code is based on the following repo:
  - PyTorch implementation of LDReg:  https://github.com/HanxunH/LDReg
  - PyTorch implementation of Faster AutoAugment: https://github.com/moskomule/dda
