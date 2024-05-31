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


class CholecSeg8k(data.Dataset):
    def __init__(self, seed=0, root='/data/gpfs/projects/punim1399/data/cholecseg8k/trainval',
                 img_folder_pattern='*/*endo.png', mask_folder_pattern='*/*endo_watershed_mask.png',
                 joint_transform=None, transform=None, label_transform=None,
                 num_classes=13, split="train", random_rotation_degree=0,
                 train_portion=0.8, **kwargs):
        self.root = root
        self.split = split
        self.id_to_trainid_unmerged = {50: 0,
                                       11: 1,
                                       21: 2,
                                       13: 3,
                                       12: 4,
                                       31: 5,
                                       23: 6,
                                       24: 7,
                                       25: 8,
                                       32: 9,
                                       22: 10,
                                       33: 11,
                                       5: 12,
                                       255: 255,
                                       0: 255}
        self.id_to_trainid = {50: 0,
                              11: 1,
                              21: 2,
                              13: 3,
                              12: 4,
                              31: 5,
                              23: 7,
                              24: 7,
                              25: 7,
                              32: 5,
                              22: 6,
                              33: 7,
                              5: 7,
                              255: 255,
                              0: 255}
        self.joint_transform = joint_transform
        self.transform = transform
        self.label_transform = label_transform
        self.num_classes = num_classes

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
        # print(os.path.join(root, img_folder_pattern))
        # print(self.imgs_path[0])

        # Build masks path and update imgs_path
        if mask_folder_pattern is not None:
            # print(os.path.join(root, mask_folder_pattern))
            self.masks_path = glob(os.path.join(root, mask_folder_pattern))
            # print(self.masks_path[0])
            # Rebuild img path
            new_imgs_path = []
            for mask_filename in self.masks_path:
                img_filename = mask_filename.replace('endo_watershed_mask.png', 'endo.png')
                # print(mask_filename, img_filename)
                if img_filename not in self.imgs_path:
                    raise('%s not found' % img_filename)
                else:
                    new_imgs_path.append(img_filename)
            self.imgs_path = new_imgs_path
            if len(self.imgs_path) != len(self.masks_path):
                raise('Mask have %d items, Images have %d items' % (len(self.masks_path), len(self.imgs_path)))
        else:
            self.masks_path = None

        # Split Train / Val
        test_size = 1 - train_portion
        if train_portion != 1.0:
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
        #     item, num_classes=num_classes) for item in self.masks_path]

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
                    # print('img:', img, 'mask:', mask)
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
        path_list = img_path.split(os.sep)
        img_name = path_list[-2] + '_' + os.path.splitext(path_list[-1])[0]
        img = Image.open(img_path).convert('RGB')
        if img_path.endswith('JPG') or img_path.endswith('jpg'):
            img = ImageOps.exif_transpose(img)

        # Open Mask
        if mask_path is None:
            return img, None, img_name
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask, np.uint8)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            # Convert Masks
            mask_mark = mask.copy()
            for k, v in self.id_to_trainid.items():
                binary_mask = (mask_mark == k)
                mask[binary_mask] = v
            mask = Image.fromarray(mask.astype(np.uint8), mode='L')

            # # Convert Masks
            # mask_id = np.empty(mask[:, :, 0], dtype=int)
            # mask_mark = mask.copy()
            # # mask_mark = mask.reshape(-1, 3)
            # for k, v in self.id_to_trainid.items():
            #     k = np.array(k)
            #     # mask_mark[(mask_mark == k).all(axis=2)] = v
            #     binary_mask = (mask_mark == k).all(axis=2)
            #     mask_id[binary_mask] = v
            # mask_id = Image.fromarray(mask_id.astype(np.uint8), mode='L')
        return img, mask, img_name

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        mask_path = self.masks_path[index] if self.masks_path is not None else None
        img, mask, img_name = self.read_images(img_path, mask_path)
        if self.centroids is not None:
            print('centroids:', self.centroids)
            centroid = random.choice(self.centroids[index])
        else:
            centroid = None
        img, mask, scale_float = self.do_transforms(img, mask, centroid=None)

        return img, mask, img_name, scale_float

    def __len__(self):
        return len(self.imgs_path)
