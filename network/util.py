import torch
from torch import nn
from network.Resnet import resnet50, resnet101
import numpy as np


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# draw different mask shape


def create_circular_mask(b, h, w, centers=[], radius=None):
    Y, X = np.ogrid[:h, :w]
    mask = np.ones((b, 1, h, w))
    for center in centers:
        # print('each center', center)
        cur_b = center[0]
        dist_from_center = np.sqrt((Y - center[2])**2 + (X-center[3])**2)
        mask[cur_b][0][dist_from_center <= radius] = 0
    return mask

#old version generate same mask for a batch
# def create_circular_mask(h, w, centers=[], radius=None):
#     Y, X = np.ogrid[:h, :w]
#     mask = np.ones((h, w))
#     for center in centers:
#         print('each center', center)
#         dist_from_center = np.sqrt((Y - center[2])**2 + (X-center[3])**2)
#         mask[dist_from_center <= radius] = 0
#     return mask


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class get_resnet(nn.Module):
    def __init__(self, trunk_name, output_stride=8, imagenet_pretrained=True, drop_rates=[0.0, 0.0, 0.0, 0.0,], c_k=7,
                 bn_eps=1.e-6):
        super(get_resnet, self).__init__()

        if trunk_name == 'resnet-50':
            resnet = resnet50(pretrained=imagenet_pretrained, drop_rates=drop_rates, bn_eps=bn_eps, c_k=c_k)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        elif trunk_name == 'resnet-101':
            resnet = resnet101(pretrained=imagenet_pretrained, drop_rates=drop_rates, bn_eps=bn_eps,  c_k=c_k)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1,
                                          resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unsupported output_stride {}'.format(output_stride)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        s2_features = x
        x = self.layer2(x)
        s4_features = x
        x = self.layer3(x)
        x = self.layer4(x)
        return s2_features, s4_features, x


def get_trunk(trunk_name, output_stride=8, imagenet_pretrained=True, drop_rates=[0.0, 0.0, 0.0, 0.0], bn_eps=1.e-5, c_k=7):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, 'Only stride8 supported right now'

    if trunk_name == 'resnet-50' or trunk_name == 'resnet-101':
        backbone = get_resnet(trunk_name, output_stride=output_stride, bn_eps=bn_eps, c_k=c_k,
                              imagenet_pretrained=imagenet_pretrained, drop_rates=drop_rates)
        s2_ch = 256
        s4_ch = -1
        high_level_ch = 2048
    else:
        raise 'unknown backbone {}'.format(trunk_name)
    return backbone, s2_ch, s4_ch, high_level_ch


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18), bn_eps=1.e-5):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          nn.BatchNorm2d(reduction_dim, eps=bn_eps),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim, eps=bn_eps),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, eps=bn_eps),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=dil,
                  padding=dil, bias=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True)
    )


class DPC(nn.Module):
    '''
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False, bn_eps=1.e-5):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch,
                                                 output_stride=output_stride, bn_eps=bn_eps)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


def make_seg_head(in_ch, out_ch, bot_ch=256):
    return nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False))
