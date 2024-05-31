from datasets.transform import joint_transforms
from datasets.transform import transforms as extended_transform
from torchvision import transforms
from .svhm_lc import SVHM_LC_Dataset
from .cholecseg8k import CholecSeg8k
from .cholec80_tool import CholecSeg8kTool
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import InterpolationMode
from .gaussian_blur import GaussianBlur

def _convert_to_rgb(image):
    return image.convert('RGB')

# Calculated by random sample 3000 samples
# See notebooks/filiter_blue_images_out_dae.ipynb
SVHM_MEAN = [0.3603, 0.1516, 0.1467]
SVHM_STD = [0.2496, 0.1636, 0.1601]


transform_options = {
    "SVHM_CH500": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((500, 500)),
            joint_transforms.RandomRotate(30),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((500, 500)),
        ]
    },
    "SVHM_CH300": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((300, 300)),
            joint_transforms.RandomRotate(30),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((300, 300)),
        ]
    },
    "SVHM_CH304": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((304, 304)),
            joint_transforms.RandomRotate(30),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((304, 304)),
        ]
    },
    "SVHM_CH500_val": {
        "train_transform": [
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((500, 500)),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((500, 500)),
        ]
    },
    "SVHM_LC480": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.Scale(480),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.Scale(480),
        ]
    },
    "SVHM_LC640": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, 
                                           contrast=0.25, 
                                           saturation=0.25, 
                                           hue=0.0),
            transforms.ToTensor()
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((360, 640)),
            # joint_transforms.Scale(640),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": [
            joint_transforms.FreeScale((360, 640)),
            # joint_transforms.Scale(640),
        ]
    },
    "SVHM_LC320": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, 
                                           contrast=0.25, 
                                           saturation=0.25, 
                                           hue=0.0),
            transforms.ToTensor()
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((180, 320)),
            # joint_transforms.Scale(640),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": [
            joint_transforms.FreeScale((180, 320)),
            # joint_transforms.Scale(640),
        ]
    },
    "SVHM_LC960": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, 
                                           contrast=0.25, 
                                           saturation=0.25, 
                                           hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((540, 960)),
            # joint_transforms.Scale(960),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((5400, 960)),
            # joint_transforms.Scale(960),
        ]
    },
    "SVHM_LC784": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, 
                                           contrast=0.25, 
                                           saturation=0.25, 
                                           hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "SVHM_LC784_finetune": {
        "train_transform": [
            transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.8, 
                            contrast=0.0, 
                            saturation=0.0, 
                            hue=0.0),
                    ], p=0.8,
                ),
            GaussianBlur(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, value='random'),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "SVHM_LC896": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, 
                                           contrast=0.25, 
                                           saturation=0.25, 
                                           hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((512, 896)),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((512, 896)),
        ]
    },
    "SVHM_LC960_val": {
        "train_transform": [
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.Scale(960),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.Scale(960),
        ]
    },
    "SVHM_LC960_pretrain": {
        "train_transform": [
            transforms.Resize((540, 960), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
    },
    "SVHM_LC896_pretrain": {
        "train_transform": [
            transforms.Resize((512, 896), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
    },
    "SVHM_LC960_pretrain_rotation": {
        "train_transform": [
            transforms.Resize((540, 960), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
    },
    "SVHM_LC640_pretrain": {
        "train_transform": [
            transforms.Resize((360, 640), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
    },
    "SVHM_LC640_pretrain_rotation": {
        "train_transform": [
            transforms.Resize((360, 640), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
    },
    "SVHM_LC960_pretrain_norm": {
        "train_transform": [
            transforms.Resize((540, 960), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHM_MEAN, SVHM_STD)
        ],
        "train_joint_transform": [],
        "test_transform": [
            transforms.ToTensor(),
            transforms.Normalize(SVHM_MEAN, SVHM_STD)
        ],
        "test_joint_transform": []
    },
    "SVHM_LC960_norm": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(SVHM_MEAN, SVHM_STD),
        ],
        "train_joint_transform": [
            joint_transforms.Scale(960),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
            transforms.Normalize(SVHM_MEAN, SVHM_STD),
        ],
        "test_joint_transform": [
            joint_transforms.Scale(960),
        ]
    },
    "SVHM_LC_simclr_pretrain": {
        "train_transform": [],
        "train_joint_transform": [],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "SVHM_LC_simclr_kornia_pretrain": {
        "train_transform": [
            transforms.Resize((540, 960)),
            _convert_to_rgb,
            transforms.ToTensor()
        ],
        "train_joint_transform": [],
        "test_transform": [
            _convert_to_rgb,
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "SVHM_LC_simclr_unet_pretrain": {
        "train_transform": [],
        "train_joint_transform": [],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((512, 896)),
        ]
    },
    "CholecSeg8k854": {
        "train_transform": [
            # transforms.Resize((360, 640), interpolation=InterpolationMode.BICUBIC),
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((480, 854)),
            # transforms.TenCrop(224),
            joint_transforms.RandomRotate(40),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((480, 854)),
            # joint_transforms.Scale(480),
        ]
    },
    "CholecSeg8k448": {
        "train_transform": [
            # transforms.Resize((360, 640), interpolation=InterpolationMode.BICUBIC),
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((256, 448)),
            # transforms.TenCrop(224),
            joint_transforms.RandomRotate(40),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((256, 448)),
            # joint_transforms.Scale(480),
        ]
    },
    "cholec80_pretrain": {
        "train_transform": [
            extended_transform.FreeScale((480, 854)),
            _convert_to_rgb,
            # transforms.Resize((540, 960), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "train_joint_transform": [
        ],
        "test_transform": [extended_transform.FreeScale((480, 854)),
                           transforms.ToTensor()],
        "test_joint_transform": [

        ]
    },
    "CholecSeg8k784": {
        "train_transform": [
            extended_transform.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0),
            transforms.ToTensor(),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
            joint_transforms.RandomRotate(40),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "CholecSeg8k784_finetune": {
        "train_transform": [
            transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.5, 
                            contrast=0.0, 
                            saturation=0.0, 
                            hue=0.0),
                    ], p=0.8,
                ),
            GaussianBlur(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, value='random'),
        ],
        "train_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
        ],
        "test_transform": [
            transforms.ToTensor(),
        ],
        "test_joint_transform": [
            joint_transforms.FreeScale((432, 784)),
        ]
    },
    "cholec80_896_pretrain": {
        "train_transform": [
            transforms.Resize((512, 896), interpolation=InterpolationMode.BICUBIC),
            # transforms.Resize((540, 960), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "train_joint_transform": [
        ],
        "test_transform": [transforms.Resize((512, 896), interpolation=InterpolationMode.BICUBIC),
                           transforms.ToTensor()],
        "test_joint_transform": [

        ]
    },
    "cholec80_tool": {
        "train_transform": [
            transforms.RandomResizedCrop((256, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "train_joint_transform": [
        ],
        "test_transform": [
            transforms.Resize((256, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ],
        "test_joint_transform": [
        ]
    },
    "ToTensor": {
        "train_transform": [transforms.ToTensor()],
        "train_joint_transform": [],
        "test_transform": [transforms.ToTensor()],
        "test_joint_transform": []
        },
}


dataset_options = {
    "SVHM_LC": lambda seed, path, transform, joint_tf, split, kwargs:
        SVHM_LC_Dataset(root=path, split=split, transform=transform, joint_transform=joint_tf,
                        seed=seed, **kwargs),
    'CholecSeg8k': lambda seed, path, transform, joint_tf, split, kwargs:
        CholecSeg8k(root=path, split=split, transform=transform, joint_transform=joint_tf,
                    seed=seed, **kwargs),
    'cholec80_tool': lambda seed, path, transform, joint_tf, split, kwargs:
        CholecSeg8kTool(root=path, split=split, transform=transform),
    "ImageFolder": lambda seed, path, transform, joint_tf, split, kwargs:
        ImageFolder(root=path,
                    transform=transform),
}
