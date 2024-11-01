import torch
import numpy as np
import json
import random
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
import glob


def get_data(group_names, data_dirs, augment_nums=None, lest_video_frames=25, mode='train'):
    data_dic_name_list = []
    speaker_name_list = []
    data_dic = defaultdict()
    speaker_video_name_list = {}

    total_frames = 0

    for data_dir, group_name, augment_num in zip(data_dirs, group_names, augment_nums):
        print('== start loading data from {} and group {} =='.format(data_dir, group_name))
        group_dir = os.path.join(data_dir, group_name)
        if mode == 'train':
            file_path = os.path.join(group_dir, 'train.txt')
        else:
            file_path = os.path.join(group_dir, 'val.txt')

        print('== start loading data from {} =='.format(file_path))

        with open(file_path,'r', encoding='utf-8') as f:
            data = f.readlines()
            data = data * augment_num

        cropped_frame_dir = os.path.join(group_dir, 'dinet_frames_cropped')
        hubert_npys_dir = os.path.join(group_dir, 'hubert_npys')

        for line in tqdm(data):
            prefix = line.strip()

            crop_frames_path = os.path.join(cropped_frame_dir, prefix)
            hubert_npy_path = os.path.join(hubert_npys_dir, prefix + '_hu.npy')

            if not os.path.exists(crop_frames_path) or not os.path.exists(hubert_npy_path):
                print('{} not exist'.format(prefix))
                continue
            else:
                # print(crop_frames_path)
                ori_image_list = get_frames(crop_frames_path)
                hubert_npy = np.load(hubert_npy_path)

                if abs(len(ori_image_list) * 2 - hubert_npy.shape[0] ) >= 8:
                    print('=====> {} hubert_npy and frames not match'.format(prefix))
                    continue

                frame_num = len(ori_image_list)

                # check frames
                if frame_num < lest_video_frames:
                    # print('=====> {} frames less than {}'.format(prefix, lest_video_frames))
                    continue

                key_str = group_name + '|' + prefix

                speaker_name = prefix.split('_')[:-1]
                speaker_name = '_'.join(speaker_name)

                if speaker_name not in speaker_name_list:
                    speaker_name_list.append(speaker_name)
                    speaker_video_name_list[speaker_name] = []

                data_dic_name_list.append(key_str)
                speaker_video_name_list[speaker_name].append(key_str)
                # roi_npy_path = os.path.join(roi_path, prefix + '.npy')

                data_dic[key_str] = [ori_image_list, hubert_npy_path, frame_num, speaker_name]
                # print(ori_image_list, hubert_npy_path, frame_num, speaker_name)
                total_frames += frame_num


    print('finish loading')

    if mode == 'train':
        random.shuffle(data_dic_name_list)

    print('finish loading')

    return data_dic_name_list, data_dic, speaker_name_list, total_frames


def get_frames(image_dir, if_return_npy=False):
    # image_list = glob.glob(os.path.join(image_dir, '*/*.jpg')) + glob.glob(os.path.join(image_dir, '*/*.png'))
    image_list = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    image_list.sort()

    return image_list


def get_badlist(files):
    alls = []
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                alls.append(line.strip())
    alls = list(set(alls))
    
    return alls


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        data_dirs = ['/data/xxx/']
        group_names = ['xxx']
        augment_nums = [1, ]

        self.data_dic_name_list, self.data_dic, self.speaker_name_list, total_frames = get_data(group_names, data_dirs, augment_nums, lest_video_frames=20, mode='train')

        print('total_frames: ', total_frames)

        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
        
        self.clip_frame_lenth = 4

        print('img_h: ', self.img_h)
        print('img_w: ', self.img_w)
        
    def get_audio_features(self, features, index):
        # print('features: ', features.shape, 'index: ', index)
        left = index - 4
        right = index + 4
        # left = index - 8
        # right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]

        if pad_left > 0:
            # auds = torch.cat([np.zeros_like(auds[:pad_left]), auds], dim=0)
            auds = np.concatenate([np.zeros([pad_left, auds.shape[1]]), auds], axis=0)
        if pad_right > 0:
            # auds = torch.cat([auds, np.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
            auds = np.concatenate([auds, np.zeros([pad_right, auds.shape[1]])], axis=0)

        # print('auds: ', auds.shape)

        return auds


    def __getitem__(self, index):
        key_str = self.data_dic_name_list[index]
        group_name, video_name = key_str.split('|')
        ori_image_list, hubert_npy_path, frame_num, speaker_name = self.data_dic[key_str]

        src_select_idx_0 = random.randint(2, frame_num - 1)
        ex_int_head = range(src_select_idx_0 - 25, src_select_idx_0 - 5)
        ex_int_tail = range(min(src_select_idx_0 + 5, frame_num - 2), min(src_select_idx_0 + 25, frame_num -1))
        ref_indexex = random.sample(list(ex_int_head) + list(ex_int_tail), 5)
        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in ref_indexex:
            reference_frame_path = ori_image_list[reference_anchor]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
            reference_frame_data_list.append(reference_frame_data)

        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        source_clip_list = []
        source_clip_mask_list = []
        hubert_speech_list = []
        reference_clip_list = []

        for i in range(self.clip_frame_lenth):
            src_select_idx = src_select_idx_0 + i - 3
            # src image (reference)
            src_img_path = ori_image_list[src_select_idx]
            source_image_data = cv2.imread(src_img_path)[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_image_mask = source_image_data.copy()

            source_image_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0
            # mouse_mask = np.zeros_like(source_image_data)
            # mouse_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 1

            ## load deep speech feature
            hubert_feature = np.load(hubert_npy_path)
            hubert_feature = self.get_audio_features(hubert_feature, src_select_idx * 2)
            
            source_clip_list.append(source_image_data)
            source_clip_mask_list.append(source_image_mask)
            hubert_speech_list.append(hubert_feature)
            reference_clip_list.append(reference_clip_data)

        source_clip = np.stack(source_clip_list, 0)
        source_clip_mask = np.stack(source_clip_mask_list, 0)
        hubert_speech_clip = np.stack(hubert_speech_list, 0)
        reference_clip = np.stack(reference_clip_list, 0)

        # # 2 tensor
        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        source_clip_mask = torch.from_numpy(source_clip_mask).float().permute(0, 3, 1, 2)
        reference_clip = torch.from_numpy(reference_clip).float().permute(0, 3, 1, 2)
        hubert_speech_clip = torch.from_numpy(hubert_speech_clip).float().permute(0, 2, 1)

        return source_clip, source_clip_mask, reference_clip, hubert_speech_clip

    def __len__(self):
        return self.length
