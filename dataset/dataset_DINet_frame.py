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
from torchvision import transforms


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
                
                print(len(ori_image_list) * 2, hubert_npy.shape[0])

                if abs(len(ori_image_list) * 2 - hubert_npy.shape[0] ) >= 8:
                    print('=====> {} hubert_npy and frames not match'.format(prefix))
                    continue

                frame_num = len(ori_image_list)

                # check frames
                if frame_num < lest_video_frames:
                    print('=====> {} frames less than {}'.format(prefix, lest_video_frames))
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
    # image_list = glob.glob(os.path.join(image_dir, '*/*.jpg')) + glob.glob(os.path.join(image_dir, '*/*.png'))    # ./xxx/00000.png
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

        data_dirs = ['/data/xxxx']
        group_names = ['xxxx',]
        augment_nums = [0, 0]

        self.data_dic_name_list, self.data_dic, self.speaker_name_list, total_frames = get_data(group_names, data_dirs, augment_nums, lest_video_frames=25, mode='train')

        print('total_frames: ', total_frames)

        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

        print('img_h: ', self.img_h)
        print('img_w: ', self.img_w)
        patch_size=8
        mask_ratio=0.1
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_patch = False

        self.aug_trans = transforms.Compose([
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05)
             transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.10, hue=0.1)
        ])
        self.aug_imgs_flags = True

    def random_masking_single_image(self, x_numpy):
        """
        对单张图像进行随机patch masking
        x_numpy: [H, W, C] numpy array
        return: [H, W, C] numpy array
        """
        H, W, C = x_numpy.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        num_mask = int(num_patches * self.mask_ratio)
        
        # 生成patch级别的mask
        mask = np.zeros(num_patches, dtype=bool)
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        mask[mask_indices] = True
        mask = mask.reshape(num_patches_h, num_patches_w)
        
        # 扩展mask到像素级别
        pixel_mask = np.repeat(np.repeat(mask, self.patch_size, axis=0), self.patch_size, axis=1)
        pixel_mask = np.expand_dims(pixel_mask, axis=-1)
        pixel_mask = np.repeat(pixel_mask, C, axis=2)
        
        # 应用mask
        x_masked = x_numpy.copy()
        x_masked[pixel_mask] = 0.0
        
        return x_masked

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
        src_select_idx = random.randint(0, frame_num -1)

        ex_int_head = range(src_select_idx - 25, src_select_idx - 5)
        ex_int_tail = range(min(src_select_idx + 5, frame_num - 2), min(src_select_idx + 25, frame_num -1))

        ref_indexex = random.sample(list(ex_int_head) + list(ex_int_tail), 5)

        # src image (reference)
        src_img_path = ori_image_list[src_select_idx]
        source_image_data = cv2.imread(src_img_path)[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
        source_image_mask = source_image_data.copy()

        source_image_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0
        mouse_mask = np.zeros_like(source_image_data)
        mouse_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 1

        ## load deep speech feature
        hubert_feature = np.load(hubert_npy_path)
        hubert_feature = self.get_audio_features(hubert_feature, src_select_idx * 2)

        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in ref_indexex:
            if reference_anchor < (-1 * frame_num + 1):
                reference_anchor = 0
            elif reference_anchor > frame_num - 1:
                reference_anchor = frame_num - 1

            reference_frame_path = ori_image_list[reference_anchor]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
            
            # 对每个reference图像单独进行mask
            if self.mask_patch and random.random() < 0.2:
                reference_frame_data = self.random_masking_single_image(reference_frame_data)

            reference_frame_data_list.append(reference_frame_data)

        if self.aug_imgs_flags:
            reference_frame_data_list.append(source_image_data)
            reference_frame_data_list.append(source_image_mask)
            aug_frame_data = np.stack(reference_frame_data_list, 0)
            # print('aug_frame_data: ', aug_frame_data.shape)
            aug_frame_data = torch.from_numpy(aug_frame_data).float().permute(0, 3, 1, 2)
            aug_frame_data_ = self.aug_trans(aug_frame_data)
            rf1, rf2, rf3, rf4, rf5, source_image_data, source_image_mask = aug_frame_data_[0], aug_frame_data_[1], aug_frame_data_[2], aug_frame_data_[3], \
                aug_frame_data_[4], aug_frame_data_[5], aug_frame_data_[6]
            reference_clip_data = torch.cat([rf1, rf2, rf3, rf4, rf5], 0)

        else:
            reference_clip_data = np.concatenate(reference_frame_data_list, 2)
            # # to tensor
            source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1)
            source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1)
            reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1)

        hubert_feature = torch.from_numpy(hubert_feature).float().permute(1,0)
        mouse_mask = torch.from_numpy(mouse_mask).bool().permute(2,0,1)

        # print('source_image_data: ', source_image_data.shape)
        # print('source_image_mask: ', source_image_mask.shape)
        # print('reference_clip_data: ', reference_clip_data.shape)

        return source_image_data, source_image_mask, reference_clip_data, hubert_feature, mouse_mask

    def __len__(self):
        return self.length
