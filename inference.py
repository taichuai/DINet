# from utils.deep_speech import DeepSpeech
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptions
from models.DINet import DINet

import numpy as np
import glob
import cv2
import torch
import subprocess
import random
from collections import OrderedDict
import librosa
import soundfile as sf
import time

from hubert import get_hubert_from_16k_speech, make_even_first_dim


def extract_frames_from_video(video_path,save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width),int(frame_height))


def get_audio_features(features, index):
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


if __name__ == '__main__':
    # load config
    opt = DINetInferenceOptions().parse_args()
    if not os.path.exists(opt.source_video_path):
        raise ('wrong video path : {}'.format(opt.source_video_path))
    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path,video_frame_dir)
    ############################################## extract deep speech feature ##############################################
    print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
    speech, sr = sf.read(opt.driving_audio_path)
    speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    
    print('extracting HuBERT feature from : {}'.format(opt.driving_audio_path))
    hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    hubert_hidden = make_even_first_dim(hubert_hidden)

    print('speech shape', speech_16k.shape, 'audio feature shape', hubert_hidden.shape)

    hubert_feature = hubert_hidden.detach().numpy()

    print('hubert_feature: ', hubert_feature.shape)

    res_frame_length = hubert_feature.shape[0] // 2
    pad_frame = True
    if pad_frame:
        hubert_feature_padding = np.pad(hubert_feature, ((4, 4), (0, 0)), mode='edge')
    else:
        hubert_feature_padding = hubert_feature

    ############################################## load facial landmark ##############################################
    print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    if not os.path.exists(opt.source_openface_landmark_path):
        raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
    video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int)
    ############################################## align frame with driving audio ##############################################
    print('aligning frames with driving audio')
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
    if len(video_frame_path_list) != video_landmark_data.shape[0]:
        raise ('video frames are misaligned with detected landmarks')

    video_frame_path_list.sort()
    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)

    if video_frame_path_list_cycle_length >= res_frame_length:
        print('res_video_frame_path_list: ', res_frame_length, 'res_video_landmark_data: ', video_landmark_data.shape)
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)

    if pad_frame:
        res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                        + res_video_frame_path_list \
                                        + [video_frame_path_list_cycle[-1]] * 2

        res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')

    else:
        res_video_frame_path_list_pad = res_video_frame_path_list
        res_video_landmark_data_pad = res_video_landmark_data

    print('res_video_frame_path_list: ', len(res_video_frame_path_list_pad), 'res_video_landmark_data: ', res_video_landmark_data_pad.shape, 'hubert_feature: ', hubert_feature.shape[0] //2)

    assert (hubert_feature_padding.shape[0] // 2) == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = len(res_video_frame_path_list_pad)

    print('pad lenth: ', pad_length)

    ############################################## load pretrained model weight ##############################################

    print('loading pretrained model from: {}'.format(opt.pretrained_clip_DINet_path))
    model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DINet_path):
        raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DINet_path))
    state_dict = torch.load(opt.pretrained_clip_DINet_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # print('k: ', k)
        # name = k[7:]  # remove module.
        name = k.replace('module.', '').replace('module', '')  # remove module.
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    ############################################## inference frame by frame ##############################################
    if not os.path.exists(opt.res_video_dir):
        os.mkdir(opt.res_video_dir)
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    if os.path.exists(res_video_path):
        os.remove(res_video_path)
    res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
    if os.path.exists(res_face_path):
        os.remove(res_face_path)

    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (resize_w, resize_h))

    crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[5:10, :, :])

    for clip_end_index in range(2, pad_length - 2, 1):
        ############################################## randomly select 5 reference images ##############################################
        print('selecting five reference images')
        ref_img_list = []
        ref_index_list = range(clip_end_index - 2, clip_end_index + 3)
        # ref_index_list = [i for i in ref_index_list]
        # random.shuffle(ref_index_list)

        # ref_index_list = random.sample(range(2, len(res_video_frame_path_list_pad) - 2), 5)

        for ref_index in ref_index_list:
            # print('ref_index: ', ref_index)
            # crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index:ref_index + 1, :, :])
            if not crop_flag:
                raise ('our method can not handle videos with large change of facial size!!')

            crop_radius_1_4 = crop_radius // 4
            # print('res_video_frame_path_list_pad: ', len(res_video_frame_path_list_pad))
            # print('ref: ', ref_index)

            ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index])[:, :, ::-1]
            ref_landmark = res_video_landmark_data_pad[ref_index, :, :]
            ref_img_crop = ref_img[
                    ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                    ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
                    :]

            ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)

        ref_video_frame = np.concatenate(ref_img_list, 2)
        ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
        # crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.05)
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index])[:, :, ::-1]
        frame_landmark = res_video_landmark_data_pad[clip_end_index, :, :]

        crop_frame_data = frame_data[
                            frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                            frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
                            :]

        crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
        crop_frame_data = crop_frame_data / 255.0
        crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                        opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)

        # hubert_tensor = torch.from_numpy(hubert_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        hubert_tensor = torch.from_numpy(get_audio_features(hubert_feature_padding, clip_end_index * 2)).permute(1, 0).unsqueeze(0).float().cuda()

        infer_time_start = time.time()

        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, hubert_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255

        print('infer time: ',time.time() - infer_time_start)

        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3,:,:]
        videowriter.write(frame_data[:, :, ::-1])

    videowriter.release()
    videowriter_face.release()
    video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')

    if os.path.exists(video_add_audio_path):
        os.remove(video_add_audio_path)

    cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
        res_video_path,
        opt.driving_audio_path,
        video_add_audio_path)

    subprocess.call(cmd, shell=True)

    print('done!')
