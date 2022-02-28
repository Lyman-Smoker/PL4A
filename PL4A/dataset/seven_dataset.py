import torch
import scipy.io
import os
import random
from opts import *
from PIL import Image
import numpy as np
from pydoc import locate
from torchvideotransforms import video_transforms, volume_transforms

def normalize(label, class_idx, upper = 100.0):

    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper)
    return norm_label

class Seven_Dataset(torch.utils.data.Dataset):
    def __init__(self, subset, multi_action, transform):
        self.subset = subset
        self.transforms = transform

        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.classes_name = classes_name

        if multi_action:
            self.class_idx_list = action_id_list
        else:
            self.class_idx = action_idx  # sport class index(from 1 begin)
        

        self.score_range = score_range
        # file path
        self.data_root = data_root     # '/home/share/AQA_7'
        self.split_path = os.path.join(self.data_root, 'Split_4', 'split_4_train_list.mat')     # /home/share/AQA_7/Split_4/split_4_train_list.mat
        self.split = scipy.io.loadmat(self.split_path)['consolidated_train_list']   # action_id, video_index, final_score
        if multi_action:
            self.split = [item.tolist() for item in self.split if item[0] in self.class_idx_list]
        else:
            self.split = self.split[self.split[:, 0] == self.class_idx].tolist()
        if self.subset == 'test':
            self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
            self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
            if multi_action:
                self.split_test = [item.tolist() for item in self.split_test if item[0] in self.class_idx_list]
            else:
                self.split_test = self.split_test[self.split_test[:, 0] == self.class_idx].tolist()
        
        # setting
        self.length = frame_length

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else:
            self.dataset = self.split.copy()

    def load_video(self, idx, action_name):
        data_path = os.path.join(self.data_root, 'frames', action_name)
        video_path = os.path.join(data_path, '%03d' % idx)
        video = [Image.open(os.path.join(video_path, 'image_%05d.jpg' % (i + 1))) for i in range(self.length)]
        return self.transforms(video)


    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        idx = int(sample_1[1])
        action_idx = int(sample_1[0])   # action 1~6
        action_name = self.classes_name[action_idx - 1]

        data = {}
        
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(idx, action_name)
            data['final_score'] = normalize(sample_1[2], action_idx, self.score_range)
            data['action_idx'] = action_idx
            return data
        else:
            # train phase
            data['video'] = self.load_video(idx, action_name)
            data['final_score'] = normalize(sample_1[2], action_idx, self.score_range)
            data['action_idx'] = action_idx
            return data

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455, 256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455, 256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Seven_Dataset(subset='test', multi_action=True, transform=test_trans)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2,
                                            shuffle=True,num_workers = int(4))
    for data in test_dataloader:
        print('video shape:', data['video'].shape)
        print('final score:', data['final_score'])
        print('action_idx:', data['action_idx'])
        break