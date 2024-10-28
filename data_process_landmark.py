from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptions

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import csv
import random
from collections import OrderedDict

CSV_HEADERS= ['frame','face_id','timestamp','confidence','success']
X = [f'x_{i}' for i in range(68)]
Y = [f'y_{i}' for i in range(68)]
CSV_HEADERS += X
CSV_HEADERS += Y


import dlib
class Croper:
    def __init__(self, path_of_lm='./asserts/shape_predictor_68_face_landmarks.dat'):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)
        self.detector = dlib.get_frontal_face_detector()

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        dets = self.detector(img_np, 1)
        if len(dets) == 0:
            return None
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = self.predictor(img_np, d)
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm



def extract_frames_from_frames(croper, frame_path, vid, save_dir):
    video_name = vid

    frames = sorted(glob.glob(os.path.join(frame_path, '*.jpg')))
    print(len(frames))
   
    video_landmark = []
    video_data = []
    
    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        frame_name = os.path.basename(fp).split('.')[0]
        frame_landmark = croper.get_landmark(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_landmark.append(frame_landmark)
        if i != int(frame_name):
            print(f'frame_name: {frame_name}, frame idx: {i}, path: {fp}')
        frame_data = [i+1, 0, 0, 0, 1]
        for r in range(68):
            frame_data.append(frame_landmark[r, 0])
        for c in range(68):
            frame_data.append(frame_landmark[c, 1])
        video_data.append(frame_data)

    with open(save_dir + '{}.csv'.format(video_name), 'w') as file:
        writer = csv.writer(file)
        data=[]
        writer.writerow(CSV_HEADERS)
        writer.writerows(video_data) 




if __name__ == '__main__':
    # load config
    source_video_path = 'asserts/training_data/split_video_25fps_frame/'
    save_dir = 'asserts/training_data/split_video_25fps_landmark_openface/'
    video_names = os.listdir(source_video_path)
    length = len(video_names)
    print(video_names)
    # video_names = [f'{i}'.zfill(4) for i in range(21, 200)]

    croper = Croper()
    bad_case_path = './bad_case.txt'
    file = open(bad_case_path,"a")
    file.write("***********************************"+'\n')

    for i, vid in enumerate(video_names):
        if os.path.exists(os.path.join(save_dir, vid + '.csv')):   #.mp4  .wav _deepspeech.txt  origin
            continue    
        print(f'==> processing {vid} {i} / {length}')
        frame_path = os.path.join(source_video_path, vid)
        try:
            extract_frames_from_frames(croper, frame_path, vid, save_dir)
        except:
            file.write(str(vid)+'\n')
            continue
        file.flush()

    file.close()
        
