import os
import numpy as np
import torch
import random
import json
from glob import glob
from torch.utils import data
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from datasets.SVHM import id_cls, color_mapping
from datasets.transform import uniform
from datasets.transform import joint_transforms


class SVHM_LC_Dataset(data.Dataset):
    def __init__(self, seed=0, root='/data/projects/punim1399/data/lc/darwin/SVHM/train',
                 img_folder_pattern='*/images/*.png', mask_folder_pattern=None,
                 joint_transform=None, transform=None, label_transform=None,
                 num_classes=35, split="train", random_rotation_degree=0,
                 train_portion=0.8, filiter_list_file=None, merge_bg=False,
                 outsource=False, finalised=False, major_classes=False, **kwargs):
        self.root = root
        self.split = split
        self.id_to_trainid = {}
        self.joint_transform = joint_transform
        self.transform = transform
        self.label_transform = label_transform
        self.num_classes = num_classes
        if not finalised:
            if not outsource:
                if self.num_classes == 25:
                    if not merge_bg:
                        self.id_to_trainid = {3: 255,
                                            4: 3,
                                            5: 4,
                                            6: 5,
                                            7: 6,
                                            8: 7,
                                            9: 8,
                                            10: 9,
                                            11: 255,
                                            12: 10,
                                            13: 255,
                                            14: 11,
                                            15: 12,
                                            16: 13,
                                            17: 14,
                                            18: 255,
                                            19: 15,
                                            20: 16,
                                            21: 17,
                                            22: 255,
                                            23: 255,
                                            24: 255,
                                            25: 18,
                                            26: 19,
                                            27: 255,
                                            28: 20,
                                            29: 21,
                                            30: 22,
                                            31: 23,
                                            32: 24,
                                            33: 255,
                                            34: 255, }
                    else:
                        self.id_to_trainid = {3: 0,
                                            4: 3,
                                            5: 4,
                                            6: 5,
                                            7: 6,
                                            8: 7,
                                            9: 8,
                                            10: 9,
                                            11: 0,
                                            12: 10,
                                            13: 0,
                                            14: 11,
                                            15: 12,
                                            16: 13,
                                            17: 14,
                                            18: 0,
                                            19: 15,
                                            20: 16,
                                            21: 17,
                                            22: 0,
                                            23: 0,
                                            24: 0,
                                            25: 18,
                                            26: 19,
                                            27: 0,
                                            28: 20,
                                            29: 21,
                                            30: 22,
                                            31: 23,
                                            32: 24,
                                            33: 0,
                                            34: 0, }
                elif self.num_classes == 26:
                    self.id_to_trainid = {3: 25,
                                        4: 3,
                                        5: 4,
                                        6: 5,
                                        7: 6,
                                        8: 7,
                                        9: 8,
                                        10: 9,
                                        11: 25,
                                        12: 10,
                                        13: 25,
                                        14: 11,
                                        15: 12,
                                        16: 13,
                                        17: 14,
                                        18: 25,
                                        19: 15,
                                        20: 16,
                                        21: 17,
                                        22: 25,
                                        23: 25,
                                        24: 25,
                                        25: 18,
                                        26: 19,
                                        27: 25,
                                        28: 20,
                                        29: 21,
                                        30: 22,
                                        31: 23,
                                        32: 24,
                                        33: 25,
                                        34: 25, }
                elif self.num_classes == 34:
                    self.id_to_trainid = {13: 12,
                                        14: 13,
                                        15: 14,
                                        16: 15,
                                        17: 16,
                                        18: 17,
                                        19: 18,
                                        20: 19,
                                        21: 20,
                                        22: 21,
                                        23: 22,
                                        24: 23,
                                        25: 24,
                                        26: 25,
                                        27: 26,
                                        28: 27,
                                        29: 28,
                                        30: 29,
                                        31: 30,
                                        32: 31,
                                        33: 32,
                                        34: 33, }
            if outsource:
                if self.num_classes == 22:
                    self.id_to_trainid = {3: 0,
                                        4: 3, 
                                        5: 4,
                                        6: 5,
                                        7: 0,
                                        8: 6,
                                        9: 7,
                                        10: 8,
                                        11: 9,
                                        12: 10,
                                        13: 11,
                                        14: 12,
                                        15: 13,
                                        16: 14,
                                        17: 15,
                                        18: 16,
                                        19: 0,
                                        20: 17,
                                        21: 18,
                                        22: 18,
                                        23: 19,
                                        24: 20,
                                        25: 21,
                                        26: 0,
                    }
                if self.num_classes == 24:
                    self.id_to_trainid = {7: 0,
                                        8: 7,
                                        9: 8,
                                        10: 9,
                                        11: 10,
                                        12: 11,
                                        13: 12,
                                        14: 13,
                                        15: 14,
                                        16: 15,
                                        17: 16,
                                        18: 17,
                                        19: 18,
                                        20: 19,
                                        21: 20,
                                        22: 20,
                                        23: 21,
                                        24: 22,
                                        25: 23,
                                        26: 255,
                    }
        if major_classes:
            print('Major classes Mode')
            self.num_classes = 20
            self.id_to_trainid = {0:255,
                                  1:0,
                                  2:1,
                                  3:255,
                                  4:2,
                                  5:3,
                                  6:4,
                                  7:5,
                                  8:6,
                                  9:255,
                                  10:255,
                                  11:7,
                                  12:8,
                                  13:9,
                                  14:255,
                                  15:255,
                                  16:10,
                                  17:11,
                                  18:12,
                                  19:255,
                                  20:13,
                                  21:14,
                                  22:255,
                                  23:15,
                                  24:16,
                                  25:17,
                                  26:18,
                                  27:19}
        if 'type' in kwargs:
            self.id_to_cls_name = id_cls.id_to_cls_name[kwargs['type']]
            color_code = color_mapping.color_maps[kwargs['type']]
            color_palette = []
            for idx, i in enumerate(color_code):
                if idx < self.num_classes:
                    for j in i:
                        color_palette.append(j)
            self.fill_colormap(color_palette)
        else:
            self.id_to_cls_name = None
            self.color_mapping = None
        self.centroids = None
        print('seed: ', seed)

        # Build images path from dirs
        self.imgs_path = glob(os.path.join(root, img_folder_pattern))

        # Filiter out samples in filiter_list
        if filiter_list_file is not None and split == 'train':
            f = open(filiter_list_file, "r")
            data = json.loads(f.read())
            target_imgs_path = data.values()
            count = 0
            for t in target_imgs_path:
                if t in self.imgs_path:
                    self.imgs_path.remove(t)
                    count += 1
            print('%d number of files removed from dataset' % count)

        # Build masks path and update imgs_path
        if mask_folder_pattern is not None:
            self.masks_path = glob(os.path.join(root, mask_folder_pattern))
            # Rebuild img path
            new_imgs_path = []
            for mask_filename in self.masks_path:
                if outsource or finalised:
                    img_filename = mask_filename.replace('aligned_id', 'images')
                else:
                    img_filename = mask_filename.replace('aligned_mask', 'images')
                if img_filename not in self.imgs_path:
                    raise('%s not found' % img_filename)
                else:
                    new_imgs_path.append(img_filename)
            self.imgs_path = new_imgs_path
            if len(self.imgs_path) != len(self.masks_path):
                raise('Mask have %d items, Images have %d items' % (len(self.masks_path), len(self.imgs_path)))
        else:
            self.masks_path = None

        # self.imgs_path.sort()

        # Split Train / Val
        test_size = 1 - train_portion
        if train_portion != 1.0:
            if self.masks_path is None:
                split_list = train_test_split(self.imgs_path, test_size=test_size, random_state=seed)
                if split == 'train':
                    self.imgs_path = split_list[0]
                elif split == 'val':
                    self.imgs_path = split_list[1]
                elif split == 'test':
                    pass
                else:
                    raise('Unknown mode %s' % split)
            else:
                split_list = train_test_split(self.imgs_path, self.masks_path, test_size=test_size, random_state=seed)
                if split == 'train':
                    self.imgs_path = split_list[0]
                    self.masks_path = split_list[2]
                elif split == 'val':
                    self.imgs_path = split_list[1]
                    self.masks_path = split_list[3]
                elif split == 'test':
                    pass
                else:
                    raise('Unknown mode %s' % split)
                if len(self.imgs_path) != len(self.masks_path):
                    raise('Incosistent number of files')

                # Compute Centroids
                # self.centroids = [uniform.class_centroids_image(
                    # item, num_classes=num_classes) for item in self.masks_path]

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        if self.color_mapping is not None:
            new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
            new_mask.putpalette(self.color_mapping)
        else:
            new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('L')
        return new_mask

    def fill_colormap(self, palette):

        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette

    def do_transforms(self, img, mask, centroid=None):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0
        if self.joint_transform is not None:
            for idx, xform in enumerate(self.joint_transform):
                if centroid is not None and isinstance(xform, joint_transforms.RandomSizeAndCrop):
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)
                # outputs = xform(img, mask)
                # if len(outputs) == 3:
                #     img, mask, scale_float = outputs
                # else:
                #     img, mask = outputs
        if self.transform is not None:
            # print('self.transform:', self.transform)
            img = self.transform(img)
        if self.label_transform is not None:
            # print('self.label_transform:', self.label_transform)
            mask = self.label_transform(mask)
        elif mask is not None:
            mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
        else:
            mask = img
        return img, mask, scale_float

    def read_images(self, img_path, mask_path):
        # Open Image
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert('RGB')
        if img_path.endswith('JPG') or img_path.endswith('jpg'):
            img = ImageOps.exif_transpose(img)

        # Open Mask
        if mask_path is None:
            return img, None, img_name
        else:
            mask = Image.open(mask_path).convert(mode='P')
            mask = np.array(mask, np.uint8)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            # Convert Masks
            mask_mark = mask.copy()
            for k, v in self.id_to_trainid.items():
                binary_mask = (mask_mark == k)
                mask[binary_mask] = v
            mask = Image.fromarray(mask.astype(np.uint8), mode='P')
        return img, mask, img_name

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        mask_path = self.masks_path[index] if self.masks_path is not None else None
        img, mask, img_name = self.read_images(img_path, mask_path)
        # if self.centroids is not None:
        #     centroid = random.choice(self.centroids[index])
        # else:
        centroid = None
        img, mask, scale_float = self.do_transforms(img, mask, centroid=centroid)

        return img, mask, img_name, scale_float

    def __len__(self):
        return len(self.imgs_path)
