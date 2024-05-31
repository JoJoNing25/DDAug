
import torch.nn.functional as F
import functools
import torch 
import torch.nn as nn
import kornia
import math
from kornia.enhance.adjust import _solarize, adjust_brightness_accumulative
from kornia.enhance.adjust import adjust_saturation_with_gray_subtraction, adjust_contrast_with_mean_subtraction
from kornia.enhance.adjust import adjust_hue
from kornia.geometry.transform import hflip
from kornia.filters.gaussian import gaussian_blur2d
from kornia.augmentation import ColorJitter, RandomResizedCrop
from kornia.enhance.adjust import _solarize 
from torch.autograd import Function
from kornia.geometry.transform import shear
from kornia.augmentation import RandomBrightness, RandomContrast, RandomHue, RandomSaturation, RandomSolarize, RandomGaussianBlur
from kornia.augmentation import RandomSolarize, RandomPosterize, RandomSharpness
def activation(x):
    x = (torch.tanh(x) + 1) / 2
    return x

    
# Stop gradient for discreate augmentation operations
# Following https://arxiv.org/pdf/1911.06987.pdf
# https://github.com/moskomule/dda/blob/master/dda/functional.py#L44
class _STE(Function):
    """ StraightThrough Estimator

    """
    @staticmethod
    def forward(ctx, input_forward, input_backward):
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
    
    
def ste(input_forward, input_backward):
    """ Straight-through estimator

    :param input_forward:
    :param input_backward:
    :return:
    """

    return _STE.apply(input_forward, input_backward).clone()

def tensor_function(func):
    # check if the input is correctly given and clamp the output in [0, 1]
    @functools.wraps(func)
    def inner(*args):
        if len(args) == 1:
            img = args[0]
            mag = None
        elif len(args) == 2:
            img, mag = args
        else:
            img, mag, kernel = args

        if not torch.is_tensor(img):
            raise RuntimeError(f'img is expected to be torch.Tensor, but got {type(img)} instead')

        if img.dim() == 3:
            img = img.unsqueeze(0)

        if torch.is_tensor(mag) and mag.nelement() != 1 and mag.size(0) != img.size(0):
            raise RuntimeError('Shape of `mag` is expected to be `1` or `B`')

        out = func(img, mag, kernel) if len(args) == 3 else func(img, mag)
        return out.clamp_(0, 1)

    return inner

def _blend_image(img1, img2, alpha):
    # blend two images
    # alpha=1 returns the original image (img1)
    alpha = alpha.view(-1, 1, 1, 1)
    return (img2 + alpha * (img1 - img2)).clamp(0, 1)

def _blur(img: torch.Tensor,
          kernel: torch.Tensor) -> torch.Tensor:
    assert kernel.ndim == 2
    c = img.size(1)
    return F.conv2d(F.pad(img, (1, 1, 1, 1), 'reflect'),
                    kernel.repeat(c, 1, 1, 1),
                    padding=0,
                    stride=1,
                    groups=c)
    
class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    
    def forward(self, x):
        return x
    
    def _get_params(self):
        return None
    
    def _sample(self):
        return (
            'None', 
            None,
            None,
        )
    
class BrightnessLayer(nn.Module):
    def __init__(self):
        super(BrightnessLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None
    
    def forward(self, x):
        if x.shape[0] == 0:
            return x
        params = activation(self.params)
        if self.training:
            x = adjust_brightness_accumulative(x, factor=params)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                if params.item() <= 1e-3 :
                    s = 1e-3
                else:
                    s = params.item()
                self.apply_fn = RandomBrightness(brightness=(1e-4, s), p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Brightness', 
            activation(self.params).tolist()
        )
    
    def _get_params(self):
        return activation(self.params).tolist()
    
    def _get_auglayer(self):
        return self.augment_op

    
    
class ContrastLayer(nn.Module):
    def __init__(self):
        super(ContrastLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None
        
    def forward(self, x):
        if x.shape[0] == 0:
            return x
        params = activation(self.params)
        if self.training:
            x = adjust_contrast_with_mean_subtraction(x, factor=params)
            x = torch.clamp(x, 0, 1)
        else: 
            if self.apply_fn is None:
                if params.item() <= 0.01:
                    s = 0.01
                else:
                    s = params.item()
                self.apply_fn = RandomContrast(contrast=(0.01, s), p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Contrast', 
            activation(self.params).tolist()
        )
    
    def _get_params(self):
        return activation(self.params).tolist()
        
    
class HueLayer(nn.Module):
    def __init__(self):
        super(HueLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        # range is [-pi, pi], use tanh * pi
        if self.training:
            params = torch.tanh(self.params) * torch.acos(torch.zeros(1)).item() * 2
            x = adjust_hue(x, factor=params)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                params = torch.tanh(self.params) / 2
                range_params = abs(params.item())
                self.apply_fn = RandomHue(hue=(-range_params, range_params), p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Hue', 
            (torch.tanh(self.params) * torch.acos(torch.zeros(1)).item() * 2).tolist()
        )
    
    def _get_params(self):
        return (torch.tanh(self.params) * torch.acos(torch.zeros(1)).item() * 2).tolist()
        
    
class SaturationLayer(nn.Module):
    def __init__(self):
        super(SaturationLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None
    def forward(self, x):
        if x.shape[0] == 0:
            return x
        # params in range of [0, 2]
        params = activation(self.params)*2
        if self.training:
            x = adjust_saturation_with_gray_subtraction(x, factor=params)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                if params.item() <= 1e-3 :
                    s = 1e-3
                else:
                    s = params.item()
                self.apply_fn = RandomSaturation(saturation=(1e-4, s), p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Saturation', 
            (activation(self.params)*2).tolist()
        )
    
    def _get_params(self):
        return (activation(self.params)*2).tolist()

    
@tensor_function
def solarize(img: torch.Tensor,
             mag: torch.Tensor) -> torch.Tensor:
    mag = mag.view(-1, 1, 1, 1)
    return ste(torch.where(img < mag, img, 1 - img), mag)

class SolarizeLayer(nn.Module):
    def __init__(self):
        super(SolarizeLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None

    def forward(self, x):
        params = activation(self.params)
        if self.training:
            x = ste(solarize(x, params), params)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                self.apply_fn = RandomSolarize(thresholds=params.item(), p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Solarize', 
            activation(self.params).tolist()
        )
    
    def _get_params(self):
        return activation(self.params).tolist()


    
class GaussianBlurLayer(nn.Module):
    def __init__(self):
        super(GaussianBlurLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(2))
        self.apply_fn = None
    def forward(self, x):
        if x.shape[0] == 0:
            return x
        sigma = F.relu(self.params)
        if self.training:
            sigma = sigma.repeat(x.shape[0], 1)
            x = gaussian_blur2d(x, kernel_size=(23, 23), sigma=sigma)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                sigma = torch.sort(sigma)[0]
                self.apply_fn = RandomGaussianBlur(kernel_size=(23, 23), sigma=sigma, p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'GaussianBlur', 
            F.relu(self.params).tolist()
        )
    def _get_params(self):
        return F.relu(self.params).tolist()
    
@tensor_function
def posterize(img: torch.Tensor,
              mag: torch.Tensor) -> torch.Tensor:
    # mag: 0 to 1
    mag = mag.view(-1, 1, 1, 1)
    with torch.no_grad():
        shift = ((1 - mag) * 8).long()
        shifted = (img.mul(255).long() << shift) >> shift
    return ste(shifted.float() / 255, mag)


class PosterizeLayer(nn.Module):
    def __init__(self):
        super(PosterizeLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None

    def forward(self, x):
        params = activation(self.params)
        if self.training:
            x = posterize(x, params)
            x = torch.clamp(x, 0, 1)
        else:
            if self.apply_fn is None:
                self.apply_fn = RandomPosterize(bits=params.item() * 8, p=1.0)
            x = self.apply_fn(x)
        return x
    
    def _sample(self):
        return (
            'Posterize', 
            activation(self.params).tolist()
        )
    
    def _get_params(self):
        return activation(self.params).tolist()

    
def _gray(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.chunk(3, dim=1)
    return 0.299 * r + 0.587 * g + 0.110 * b

class GrayLayer(nn.Module):
    def __init__(self):
        super(GrayLayer, self).__init__()
    
    def forward(self, x):
        return _gray(x).repeat(1, 3, 1, 1)

    def _sample(self):
        return (
            'Gray', 
            None,
        )
    
    def _get_params(self):
        return None


def get_sharpness_kernel(device):
    kernel = torch.ones(3, 3, device=device)
    kernel[1, 1] = 5
    kernel /= 13
    return kernel

@tensor_function
def sharpness(img, mag):
    kernel = get_sharpness_kernel(img.device)
    return _blend_image(img, _blur(img, kernel), 1 - mag)

class SharpnessLayer(nn.Module):
    def __init__(self):
        super(SharpnessLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(1))
        self.apply_fn = None

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        if self.training:
            return sharpness(x, activation(self.params))
        else:
            if self.apply_fn is None:
                params = activation(self.params)
                if params.item() <= 1e-3:
                    s = 1e-3
                else:
                    s = params.item()
                self.apply_fn = RandomSharpness(sharpness=(1e-4, s), p=1.0)
            return self.apply_fn(x)

    def _sample(self):
        return (
            'Sharpness', 
            activation(self.params).tolist(),
        )
    
    def _get_params(self):
        return activation(self.params).tolist()

class ShearLayer(nn.Module):
    def __init__(self):
        super(ShearLayer, self).__init__()
        self.params = nn.Parameter(torch.rand(2))
    
    def forward(self, x):
        params = activation(self.params)
        params = params.repeat(x.shape[0], 1)
        x = shear(x, params)
        x = torch.clamp(x, 0, 1)
        return x
    
    def _sample(self):
        return (
            'Shear', 
            activation(self.params).tolist()
        )
    
    def _get_params(self):
        return activation(self.params).tolist()


# Register
OPS = {
  'None': IdentityLayer,
  'Brightness' : BrightnessLayer,
  'Contrast' : ContrastLayer,
  'Hue' : HueLayer,
  'Saturation' : SaturationLayer,
  'Solarize' : SolarizeLayer,
  'GaussianBlur' : GaussianBlurLayer,
  'Posterize': PosterizeLayer,
  'Gray': GrayLayer,
  'Sharpness': SharpnessLayer,
}

OPS_2 = {
  'None': IdentityLayer,
  'Brightness' : BrightnessLayer,
  'Contrast' : ContrastLayer,
  'Hue' : HueLayer,
  'Saturation' : SaturationLayer,
  'Solarize' : SolarizeLayer,
  'GaussianBlur' : GaussianBlurLayer,
  'Posterize': PosterizeLayer,
  'Gray': GrayLayer,
  'Sharpness': SharpnessLayer,
}

class AugLayer(nn.Module):
    def __init__(self, temperature=0.1, sample_mode='argmax', search_space='S1'):
        super(AugLayer, self).__init__()
        self._ops = nn.ModuleList()
        if search_space == 'S1':
            space = OPS
        else:
            space = OPS_2
        for op in space:
            augment_op = OPS[op]()
            self._ops.append(augment_op)
        self.p = nn.Parameter(torch.rand(len(self._ops)))
        self.temperature = temperature
        self.sample_mode = sample_mode
        
    def forward(self, x):
        weights = F.softmax(self.p / self.temperature, dim=0)
        if self.training:
            x = sum([w * op(x) for w, op in zip(weights, self._ops)])
        else:
            if self.sample_mode == 'argmax':
                p = weights.max().to(x.device)
                p = p.repeat(x.shape[0])
                r = torch.rand_like(p, device=x.device)
                op_idx = weights.argmax().to(x.device)
                x[r<p, :, :, :] = self._ops[op_idx](x[r<p, :, :, :])    
            elif self.sample_mode == 'weighted':
                dist = torch.distributions.categorical.Categorical(probs=weights)
                for i in range(x.shape[0]):
                    op_idx = dist.sample().to(x.device)
                    x[i, :, :, :] = self._ops[op_idx](x[i, :, :, :].unsqueeze(0))
                    
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, 0, 1)
        return x
    
    def _sample(self):
        p = F.softmax(self.p / self.temperature, dim=0)
        idx = torch.argmax(p)
        return ['Prob: {:4f}'.format(p.max().item()), self._ops[idx]._sample()]
        
        
class AugPolicy(nn.Module):
    def __init__(self, layers=5, temperature=0.5, sample_mode='argmax', search_space='S1'):
        super(AugPolicy, self).__init__()
        layers = [AugLayer(temperature=temperature, sample_mode=sample_mode, search_space=search_space) for _ in range(layers)]
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.layers(x)
        
    def _sample(self):
        return [l._sample() for l in self.layers]
    