import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import functional as TF

import numpy as np
from PIL import ImageOps
from PIL import ImageFilter

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

from PIL import ImageFilter
import numpy as np
from kornia.augmentation import ColorJitter, RandomHorizontalFlip, RandomGrayscale, RandomResizedCrop


class GaussianBlur(object):
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Scale(object):
    """
    Scale image such that longer side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        if w > h:
            long_edge = w
        else:
            long_edge = h

        if long_edge == self.size:
            return img

        scale = self.size / long_edge
        target_w = int(w * scale)
        target_h = int(h * scale)
        target_size = (target_w, target_h)

        return img.resize(target_size, Image.BILINEAR)


class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing
            any artifacts.

    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.

        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample = TF.rotate(sample, self.angle)
        return sample




class Jigsaw(object):
    """Implementation of Jigsaw image augmentation, inspired from PyContrast library.

    Generates n_grid**2 random crops and returns a list.

    This augmentation is instrumental to PIRL.

    Attributes:
        n_grid:
            Side length of the meshgrid, sqrt of the number of crops.
        img_size:
            Size of image.
        crop_size:
            Size of crops.
        transform:
            Transformation to apply on each crop.

    Examples:
        >>> from lightly.transforms import Jigsaw
        >>>
        >>> jigsaw_crop = Jigsaw(n_grid=3, img_size=255, crop_size=64, transform=transforms.ToTensor())
        >>>
        >>> # img is a PIL image
        >>> crops = jigsaw_crops(img)
    """

    def __init__(self, n_grid=3, img_size=255, crop_size=64, transform=transforms.ToTensor()):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size
        self.transform = transform

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        """Performs the Jigsaw augmentation
        Args:
            img:
                PIL image to perform Jigsaw augmentation on.
        Returns:
            Torch tensor with stacked crops.
        """
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        crops = [Image.fromarray(crop) for crop in crops]
        crops = torch.stack([self.transform(crop) for crop in crops])
        crops = crops[np.random.permutation(self.n_grid ** 2)]
        return crops


class RandomSolarization(object):
    """Implementation of random image Solarization.
    Utilizes the integrated image operation `solarize` from Pillow. Solarization
    inverts all pixel values above a threshold (default: 128).
    Attributes:
        probability:
            Probability to apply the transformation
        threshold:
            Threshold for solarization.
    """

    def __init__(self, prob: float = 0.5,
                 threshold: int = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample):
        """Solarizes the given input image
        Args:
            sample:
                PIL image to which solarize will be applied.
        Returns:
            Solarized image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # return solarized image
            return ImageOps.solarize(sample, threshold=self.threshold)
        # return original image
        return sample


class BaseCollateFunction(nn.Module):
    """Base class for other collate implementations.
    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.
    """

    def __init__(self, transform: torchvision.transforms.Compose):

        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.
            Args:
                batch:
                    A batch of tuples of images, labels, and filenames which
                    is automatically provided if the dataloader is built from
                    a LightlyDataset.
            Returns:
                A tuple of images, labels, and filenames. The images consist of
                two batches corresponding to the two transformations of the
                input images.
            Examples:
                >>> # define a random transformation and the collate function
                >>> transform = ... # some random augmentations
                >>> collate_fn = BaseCollateFunction(transform)
                >>>
                >>> # input is a batch of tuples (here, batch_size = 1)
                >>> input = [(img, 0, 'my-image.png')]
                >>> output = collate_fn(input)
                >>>
                >>> # output consists of two random transforms of the images,
                >>> # the labels, and the filenames in the batch
                >>> (img_t0, img_t1), label, filename = output
        """
        batch_size = len(batch)
        # list of transformed images
        transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
                      for i in range(2 * batch_size)]
        # list of labels
        # labels = torch.LongTensor([item[1] for item in batch])
        labels = None

        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0)
        )

        return transforms, labels, fnames


class ImageCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for images.
    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.
    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings.
    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(self,
                 scale_size: int = 480,
                 input_size: int = 64,
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.7,
                 cj_contrast: float = 0.7,
                 cj_sat: float = 0.7,
                 cj_hue: float = 0.2,
                 min_scale: float = 0.15,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 normalize: dict = imagenet_normalize):

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

        transform = [Scale(size=scale_size),
                     T.RandomResizedCrop(size=input_size,
                                         scale=(min_scale, 1.0)),
                     RandomRotate(prob=rr_prob),
                     T.RandomHorizontalFlip(p=hf_prob),
                     T.RandomVerticalFlip(p=vf_prob),
                     T.RandomApply([color_jitter], p=cj_prob),
                     T.RandomGrayscale(p=random_gray_scale),
                     GaussianBlur(p=gaussian_blur),
             T.ToTensor()
        ]

        # if normalize:
        #     transform += [
        #      T.Normalize(
        #         mean=normalize['mean'],
        #         std=normalize['std'])
        #      ]

        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)



class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class TwoViewCollateFunction(nn.Module):
    def __init__(self, transform1, transform2):
        super(TwoViewCollateFunction, self).__init__()
        self.transform1 = transform1
        self.transform2 = transform2
        
    def forward(self, batch):

        batch_size = len(batch)

        # list of transformed images
        transforms1 = [self.transform1(batch[i][0]).unsqueeze_(0)
                       for i in range(batch_size)]
        
        transforms2 = [self.transform2(batch[i][0]).unsqueeze_(0)
                       for i in range(batch_size)]
        
        # tuple of transforms
        transforms = (
            torch.cat(transforms1, 0),
            torch.cat(transforms2, 0)
        )

        return transforms, None

class BYOLCollateFunction(TwoViewCollateFunction):
    def __init__(self,
                 scale_size: int = 480,
                 input_size = (432, 784),
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.4,
                 cj_contrast: float = 0.4,
                 cj_sat: float = 0.2,
                 cj_hue: float = 0.1,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 hf_prob: float = 0.5,
                 **kwargs
                 ):
        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

        transform1 = [
            Scale(size=scale_size),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=1.0),
            T.ToTensor(),
        ]
        transform2 = [
            Scale(size=scale_size),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
        ]
        transform1 = T.Compose(transform1)
        transform2 = T.Compose(transform2)
        super(BYOLCollateFunction, self).__init__(transform1, transform2)


class SimCLRCollateFunction(TwoViewCollateFunction):
    def __init__(self,
                 scale_size: int = 960,
                 input_size = (432, 784),
                 cj_prob: float = 0.8,
                 cj_strength: float = 1.0,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 hf_prob: float = 0.5):

        color_jitter = T.ColorJitter(
            cj_strength * 0.8, cj_strength * 0.8, cj_strength * 0.8 , cj_strength * 0.2
        )
        transform1 = [
            Scale(size=scale_size),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=0.5),
            T.ToTensor(),
        ]
        transform1 = T.Compose(transform1)
        super(SimCLRCollateFunction, self).__init__(transform1, transform1)


class SimCLRNoCropCollateFunction(TwoViewCollateFunction):
    def __init__(self,
                 input_size = (432, 784),
                 cj_prob: float = 0.8,
                 cj_strength: float = 1.0,
                 random_gray_scale: float = 0.2,
                 hf_prob: float = 0.5):

        color_jitter = T.ColorJitter(
            cj_strength * 0.8, cj_strength * 0.8, cj_strength * 0.8 , cj_strength * 0.2
        )
        transform1 = [
            T.Resize(size=input_size),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=0.5),
            T.ToTensor(),
        ]
        transform1 = T.Compose(transform1)
        super(SimCLRNoCropCollateFunction, self).__init__(transform1, transform1)



class SimCLRCustomCollateFunction(TwoViewCollateFunction):
    def __init__(self,
                 scale_size: int = 960,
                 input_size = (432, 784),
                 cj_prob: float = 0.8,
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0,
                 hue: float = 0.1,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 hf_prob: float = 0.5):
        
        
        color_jitter = T.ColorJitter(
            brightness, contrast, saturation, hue
        )
        transform1 = [
            Scale(size=scale_size),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=0.5),
            T.ToTensor(),
        ]
        transform1 = T.Compose(transform1)
        super(SimCLRCustomCollateFunction, self).__init__(transform1, transform1)



