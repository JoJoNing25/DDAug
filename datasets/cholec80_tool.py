import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from PIL import Image
from sklearn.model_selection import train_test_split


class CholecSeg8kTool(data.Dataset):
    def __init__(self, seed=0, root='/scratch/punim1399/cholec80', split='train',
                 joint_transform=None, transform=None, label_transform=None,
                 train_videos=[
                     'video01', 'video02', 'video03', 'video04', 'video05', 
                     'video06', 'video07', 'video08', 'video09', 'video10', 'video11', 
                     'video12', 'video13', 'video14', 'video15', 'video16', 'video17', 
                     'video18', 'video19', 'video20', 'video21', 'video22', 'video23', 
                     'video24', 'video25', 'video26', 'video27', 'video28', 'video29', 
                     'video30', 'video31', 'video32', 'video33', 'video34', 'video35',
                     'video36', 'video37', 'video38', 'video39', 'video40', 'video41', 
                     'video42', 'video43', 'video44', 'video45', 'video46', 'video47', 
                     'video48', 'video49', 'video50', 'video51', 'video52', 'video53', 
                     'video54', 'video55', 'video56', 'video57', 'video58', 'video59', 
                     'video60', 'video61', 'video62', 'video63', 'video64', 'video65', 
                     'video66', 'video67', 'video68', 'video69', 'video70'],
                 test_videos=['video71', 'video72', 'video73', 'video74', 'video75', 
                              'video76', 'video77', 'video78', 'video79', 'video80'],
                 train_portion=0.8, **kwargs):
        self.seed = seed
        self.root = root
        self.split = split
        self.joint_transform = joint_transform
        self.transform = transform
        self.label_transform = label_transform
        self.train_videos = train_videos
        self.test_videos = test_videos
        self.train_portion = train_portion

        # Build the dataset
        self.imgs = []
        self.labels = []
        if split == 'train' or split == 'val':
            self.build_dataset(self.train_videos)
        elif split == 'test':
            self.build_dataset(self.test_videos)
        else:
            raise('Unknown Split')
        
        # split train into train and val subsets
        if split == 'train' or split=='val' and train_portion < 1.0:
            test_size = 1 - train_portion
            split_list = train_test_split(self.imgs, self.labels, test_size=test_size, random_state=seed)
            if split == 'train':
                self.imgs = split_list[0]
                self.labels = split_list[2]
            elif split == 'val':
                self.imgs = split_list[1]
                self.labels = split_list[3]

    def build_dataset(self, video_list):
        for video_id in video_list:
            label_file_path = os.path.join(self.root, 'tool_annotations', video_id) + '-tool.txt'
            img_path = os.path.join(self.root, 'videos', video_id)
            df = pd.read_csv(label_file_path, sep='\t')
            for index, row in df.iterrows():
                frame_id = row[0]
                one_hot_label = np.array(row[1:])
                img_name = os.path.join(img_path, "{:08}".format(frame_id) + '.png')
                if not os.path.exists(img_name):
                    # print('Image not found: ', img_name)
                    pass
                else:
                    self.imgs.append(img_name)
                    self.labels.append(one_hot_label)
    
    def read_images(self, img_path):
        # Open Image
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.read_images(img_path)
        label = self.labels[index]
        label = torch.tensor(label).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.imgs)


# # For test
# train_dataset = CholecSeg8kTool()
# print(train_dataset.__len__())
# print(train_dataset[0])
# val_dataset = CholecSeg8kTool(split='val')
# print(val_dataset.__len__())
# test_dataset = CholecSeg8kTool(split='test')
# print(test_dataset.__len__())
