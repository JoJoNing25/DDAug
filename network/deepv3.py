import torch
import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from torch import nn
from PIL import Image
from network.util import get_aspp, get_trunk, make_seg_head, Upsample, initialize_weights, create_circular_mask
from timm.models.layers import trunc_normal_


class DeepV3PlusDAE(nn.Module):
    def __init__(self, num_classes, trunk='wrn38', mask_ratio=0.75, mask_patch_size=16,
                 use_dpc=False, init_all=False, imagenet_pretrained=False,
                 norm_pix_loss=False, dynamic_patch=False, grey_scale_prob=0.0,
                 random_rotation_degree=0, dynamic_ratio=False,
                 mask_shape='square', color_mask=False, color_pixel_mask=False):
        super(DeepV3PlusDAE, self).__init__()
        self.norm_pix_loss = norm_pix_loss
        self.dynamic_patch = dynamic_patch
        self.grey_scale_prob = grey_scale_prob
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.random_rotation_degree = random_rotation_degree
        self.mask_shape = mask_shape
        self.color_mask = color_mask
        self.color_pixel_mask = color_pixel_mask
        self.dynamic_ratio = dynamic_ratio
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk, imagenet_pretrained=imagenet_pretrained,
                                                                bn_eps=1.e-6)
        if not self.random_rotation_degree:
            self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                              bottleneck_ch=256,
                                              output_stride=8,
                                              bn_eps=1.e-6,
                                              dpc=use_dpc)
            self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
            self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
            self.final = nn.Sequential(
                nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-6),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-6),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
            initialize_weights(self.final)
        else:
            self.fc = nn.Linear(high_level_ch, 1)

    def _get_layer_ids(self):
        layer_ids = []
        for block_id, block in enumerate([
                self.backbone.layer0,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4]):
            name = 'backbone.layer%d' % (block_id)
            layer_ids.append(name)

        layer_ids.append('aspp')
        layer_ids.append('bot')
        layer_ids.append('final')

        return layer_ids

    def _grey_scale(self, x):
        # Convert Input tensor x to grey scale and back to rgb according to grey_scale_prob
        p = np.random.uniform(1.e-8, 1)
        if p < self.grey_scale_prob:
            x = x[:, 0, :, :] * 0.2989 + x[:, 1, :, :] * 0.5870 + x[:, 2, :, :] * 0.1140
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return x

    def _random_rotation(self, x):
        x_rotated = []
        rotation_degrees = []
        for i in range(x.shape[0]):
            rotate_degree = random.uniform(-1 * self.random_rotation_degree, self.random_rotation_degree)
            x_rotated.append(TF.rotate(x[i], rotate_degree, Image.BILINEAR))
            rotation_degrees.append(rotate_degree)
        rotated = torch.stack(x_rotated)
        degrees = torch.tensor(rotation_degrees)
        # print('rotated input shape:', rotated.shape, 'rotation degrees', degrees.shape)
        return rotated.to(x.device), degrees.to(x.device)

    def _get_rotation_loss(self, x, rotation):
        fn = nn.MSELoss(reduction='none')
        loss = fn(x.view(-1), rotation).mean()
        return loss

    def _get_dae_loss(self, x, x_re, mask):
        fn = nn.MSELoss()
        target = x
        if self.norm_pix_loss:
            mean = target.mean(dim=0, keepdim=True)
            var = target.var(dim=0, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = fn(x_re, target)
        return loss

    def _get_x_mask(self, x):
        x = self._grey_scale(x)
        b, c, h, w = x.shape
        # print('input shape', x.shape)
        if self.dynamic_patch:
            mask_patch_size = random.randint(self.mask_patch_size[0], self.mask_patch_size[1])
        else:
            # if circle, radius
            mask_patch_size = self.mask_patch_size
        if self.dynamic_ratio:
            mask_ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        else:
            mask_ratio = self.mask_ratio
        if self.color_mask:
            color_mask = torch.rand(b, c, h//mask_patch_size, w//mask_patch_size)
            color_mask = color_mask.type(torch.FloatTensor).to(x.device)
            if color_mask.shape[2] != h and color_mask.shape[3] != w:
                color_mask = F.upsample(color_mask, size=(h, w), mode='nearest')
        if self.color_pixel_mask:
            color_mask = torch.rand(b, 3, h, w).to(x.device)
        if self.mask_shape == 'square':
            mask = torch.rand(b, 1, h//mask_patch_size, w//mask_patch_size).to(x.device)
            mask = mask > mask_ratio  # larger than ratio become 1, 0 is center
            mask = mask.type(torch.FloatTensor).to(x.device)
            if mask.shape[2] != h and mask.shape[3] != w:
                mask = F.upsample(mask, size=(h, w), mode='nearest')
            # x_masked = x * mask
        elif self.mask_shape == 'circle':
            mask = torch.rand(b, 1, h//mask_patch_size, w//mask_patch_size).to(x.device)
            mask = mask > mask_ratio  # larger than ratio become 1, 0 is mask center
            mask = mask.type(torch.FloatTensor).to(x.device)
            jo_mask = mask.detach().cpu().numpy()
            jo_centers = np.argwhere(jo_mask == 0)
            for i, center in enumerate(jo_centers):
                center[2] = center[2] * mask_patch_size + mask_patch_size//2
                center[3] = center[3] * mask_patch_size + mask_patch_size//2
            if mask.shape[2] != h and mask.shape[3] != w:
                mask = create_circular_mask(b, h, w, jo_centers, radius=mask_patch_size//2)
                mask = torch.tensor(mask).type(torch.FloatTensor).to(x.device)  # .view(1, 1, h, w)

        elif self.mask_shape == 'None':
            mask = torch.ones(b, 1, h, w).type(torch.FloatTensor).to(x.device)

        if self.color_mask:
            x_masked = x * mask + (1-mask) * color_mask
            return x_masked, (1-mask) * color_mask + mask
        else:
            x_masked = x * mask
        return x_masked, mask
    # previous version square mask
    # def _get_x_mask(self, x):
    #     x = self._grey_scale(x)
    #     b, c, h, w = x.shape
    #     if self.dynamic_patch:
    #         mask_patch_size = random.randint(self.mask_patch_size[0], self.mask_patch_size[1])
    #     else:
    #         mask_patch_size = self.mask_patch_size
    #     mask = torch.rand(b, 1, h//mask_patch_size, w//mask_patch_size).to(x.device)
    #     mask = mask > self.mask_ratio
    #     mask = mask.type(torch.FloatTensor).to(x.device)
    #     if mask.shape[2] != h and mask.shape[3] != w:
    #         mask = F.upsample(mask, size=(h, w), mode='nearest')
    #     x_masked = x * mask
    #     return x_masked, mask

    def forward(self, img):
        if self.training:
            if self.random_rotation_degree:
                x, rotation_degree = self._random_rotation(img)
            else:
                x, mask = self._get_x_mask(img)
        else:
            x = img
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        if self.random_rotation_degree:
            x = final_features.mean(dim=[2, 3])
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            # print('before loss calculation:', out.shape, type(out))
            loss = self._get_rotation_loss(out, rotation_degree)
            return loss, out, rotation_degree
        lid_feature = final_features
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        if not self.norm_pix_loss:
            out = torch.sigmoid(out)
        if self.training:
            loss = self._get_dae_loss(img, out, mask)
            return loss, out, mask
        return out


class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    active_learning = ['feature', 'entropy']
    layer = None if entropy else customed
    """

    def __init__(self, num_classes, trunk='wrn38', criterion=None, drop_rates=[0.0, 0.0, 0.0, 0.0],
                 use_dpc=False, init_all=False, imagenet_pretrained=True, fix_trunk=False, c_k=7,
                 active_learning=None, layer=None):
        super(DeepV3Plus, self).__init__()
        self.active_learning = active_learning
        self.layer = layer
        if self.active_learning == 'feature':
            self.lid_list = []
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk, imagenet_pretrained=imagenet_pretrained,
                                                                bn_eps=1.e-6, c_k=c_k,
                                                                drop_rates=drop_rates)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          bn_eps=1.e-6,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-6),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-6),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final)
        if fix_trunk:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_layer_ids(self):
        layer_ids = []
        for block_id, block in enumerate([
                self.backbone.layer0,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4]):
            name = 'backbone.layer%d' % (block_id)
            layer_ids.append(name)

        layer_ids.append('aspp')
        layer_ids.append('bot')
        layer_ids.append('final')

        return layer_ids

    def finetune(self):
        trunc_normal_(self.final[-1].weight, std=2.e-5)

    def forward_features(self, batch_imgs):
        x = batch_imgs
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        if self.layer == 'backbone':
            # print('feature shape after {} layer: {}'.format(self.layer, final_features.shape))
            if len(final_features.shape) > 2:
                selected_features = final_features.mean(-1).mean(-1)
            else:
                selected_features = final_features
            # print('feature shape after mean:', selected_features.shape)
            return selected_features
        aspp = self.aspp(final_features)
        if self.layer == 'aspp':
            # print('feature shape after {} layer: {}'.format(self.layer, aspp.shape))
            if len(aspp.shape) > 2:
                selected_features = aspp.mean(-1).mean(-1)
            else:
                selected_features = aspp
            # print('aspp layer feature shape mean:', selected_features.shape)
            return selected_features
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        if self.layer == 'cat_s4':
            # print('feature shape after {} layer: {}'.format(self.layer, cat_s4.shape))
            if len(cat_s4.shape) > 2:
                selected_features = cat_s4.mean(-1).mean(-1)
            else:
                selected_features = cat_s4
            # print('cat_s4 layer feature shape mean:', selected_features.shape)
            return selected_features
        final = self.final(cat_s4)
        if self.layer == 'final':
            # print('feature shape after {} layer: {}'.format(self.layer, final.shape))
            if len(final.shape) > 2:
                selected_features = final.mean(-1).mean(-1)
            else:
                selected_features = final
            # print('final layer feature shape mean:', selected_features.shape)
            return selected_features
        if self.active_learning == 'entropy':
            out = Upsample(final, x_size[2:])
            predict_prob = torch.nn.functional.softmax(out, dim=1).cpu().data
            m = torch.nn.LogSoftmax(dim=1)
            predict_ll = m(out).cpu().data
            entropy = - predict_prob * predict_ll
            output = torch.sum(entropy, dim=1)
            # print('shape after sum', output.shape)
            output = torch.mean(output, dim=(1, 2))
            # print('shape after mean', output.shape)
            return output

    def forward(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        # print(s2_features.shape)
        # print(final_features.shape)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        # print(cat_s4.shape)
        final = self.final(cat_s4)
        # print(final.shape)
        out = Upsample(final, x_size[2:])

        if self.training:
            return out

        return {'pred': out}


def DeepV3PlusSRNX50(num_classes, criterion, cal_lid):
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion)


def DeepV3PlusR50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3PlusR101(num_classes, criterion=None, cal_lid=None, dae=None, imagenet_pretrained=True):
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion,
                      cal_lid=cal_lid, dae=dae, imagenet_pretrained=imagenet_pretrained)


def DeepV3PlusSRNX101(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion)


def DeepV3PlusW38(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3PlusW38I(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                      init_all=True)


def DeepV3PlusX71(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='xception71', criterion=criterion)


def DeepV3PlusEffB4(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='efficientnet_b4',
                      criterion=criterion)


class DeepV3(nn.Module):
    """
    DeepLabV3 with various trunks supported
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None,
                 use_dpc=False, init_all=False, output_stride=8):
        super(DeepV3, self).__init__()
        self.criterion = criterion

        self.backbone, _s2_ch, _s4_ch, high_level_ch = \
            get_trunk(trunk, output_stride=output_stride)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=output_stride,
                                          dpc=use_dpc)
        self.final = make_seg_head(in_ch=aspp_out_ch, out_ch=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        _, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        final = self.final(aspp)
        out = Upsample(final, x_size[2:])

        if self.training:
            return out
            # assert 'gts' in inputs
            # gts = inputs['gts']
            # return self.criterion(out, gts)
        return {'pred': out}


def DeepV3R50(num_classes, criterion):
    return DeepV3(num_classes, trunk='resnet-50', criterion=criterion)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

