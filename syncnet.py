import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torch import optim
import random
import argparse
import tqdm
from collections import defaultdict
import glob


def get_data(group_names, data_dirs, augment_nums=None, lest_video_frames=25, mode='train', bad_file_list=None):
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

        # crop_frames_path = os.path.join(os.path.dirname(file_path), '3dmm_crop_frames')
        cropped_frame_dir = os.path.join(group_dir, 'dinet_frames_cropped2')
        hubert_npys_dir = os.path.join(group_dir, 'hubert_npys')
        # hubert_npys_dir = os.path.join(group_dir, 'wavlm_npys')

        for line in tqdm.tqdm(data):
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
                if bad_file_list is not None and key_str in bad_file_list:
                    # print('====bad case, skip====')
                    continue

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



class Dataset(object):
    def __init__(self, dataset_dir, is_train=True):
        data_dirs = ['/data1', '/data2']
        group_names = ['aaa', 'bbb']
        augment_nums = [1, 2]

        if is_train:
            self.data_dic_name_list, self.data_dic, self.speaker_name_list, total_frames = get_data(group_names, data_dirs, augment_nums, lest_video_frames=20, mode='train')
        else:
            self.data_dic_name_list, self.data_dic, self.speaker_name_list, total_frames = get_data(group_names, data_dirs, augment_nums, lest_video_frames=20, mode='val')

        print('total_frames: ', total_frames)

        mouth_region_size = 256

        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

        print('img_h: ', self.img_h)
        print('img_w: ', self.img_w)

        # ------------------------------------------------
        
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


    def __len__(self):
        return self.length


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

    def __getitem__(self, idx):
        key_str = self.data_dic_name_list[idx]
        # print('key_str: ', key_str)
        group_name, video_name = key_str.split('|')
        ori_image_list, hubert_npy_path, frame_num, speaker_name = self.data_dic[key_str]
        src_select_idx = random.randint(0, frame_num -1)

        # src image (reference)
        src_img_path = ori_image_list[src_select_idx]
        source_image_data = cv2.imread(src_img_path)[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
        source_image_mouth = source_image_data.copy()[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:]
        # source_image_mouth[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0

        ## load deep speech feature
        hubert_feature = np.load(hubert_npy_path)
        hubert_feature = self.get_audio_features(hubert_feature, src_select_idx * 2)

        # # to tensor
        source_image_mouth = torch.from_numpy(source_image_mouth).float().permute(2,0,1)
        audio_feat = torch.from_numpy(hubert_feature).float()

        # ----------------------------------------------------------------------
        y = torch.ones(1).float()
        
        # print('source_image_mouth: ', source_image_mouth.shape)
        # print('audio_feat: ', audio_feat.shape)

        return source_image_mouth, audio_feat, y

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# class AudioFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(AudioFeatureExtractor, self).__init__()
#         self.layer1 = ResidualBlock(32, 32)
#         self.layer2 = ResidualBlock(32, 64)
#         self.layer3 = ResidualBlock(64, 128)
#         self.layer4 = ResidualBlock(128, 256)
#         self.layer5 = ResidualBlock(256, 256)
#         self.layer6 = ResidualBlock(256, 256)
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, 512)

#     def forward(self, x):
#         print('x: ', x.shape)
#         x = x.permute(0, 2, 1)  # Change to [batch_size, 768, 16] for Conv1d
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.global_avg_pool(x)  # [batch_size, 256, 1]
#         x = x.view(x.size(0), -1)  # Flatten to [batch_size, 256]
#         x = self.fc(x)  # [batch_size, 512]
#         return x

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch_size, channels, length]
        attention_weights = self.attention_layer(x.permute(0, 2, 1)).squeeze(-1)  # [batch_size, length]
        attention_weights = torch.softmax(attention_weights, dim=-1)  # Normalize weights
        x = torch.bmm(attention_weights.unsqueeze(1), x.permute(0, 2, 1))  # [batch_size, 1, channels]
        return x.squeeze(1)  # [batch_size, channels]

class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
        self.layer1 = ResidualBlock(768, 256)
        self.layer2 = ResidualBlock(256, 256)
        self.layer3 = ResidualBlock(256, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 64)
        self.layer6 = ResidualBlock(64, 64)
        self.attention_fusion = AttentionFusion(64)
        self.fc = nn.Linear(64, 512)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to [batch_size, 768, 16] for Conv1d
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.attention_fusion(x)  # [batch_size, 64]
        # x = self.ac(self.fc(x))  # [batch_size, 512]
        x = self.fc(x)  # [batch_size, 512]

        return x


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.global_avg_pool(x)  # [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 512]
        # x = self.ac(self.fc(x))  # [batch_size, 512]
        x = self.fc(x)  # [batch_size, 512]

        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


# def init_weights2(m):
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight)  # 使用正交初始化
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)


class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            )

        self.audio_encoder = AudioFeatureExtractor()
        self.apply(init_weights)


    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        # print('face_sequences shape: ', face_sequences.shape, 'audio_sequences shape: ', audio_sequences.shape)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        # print('face_embedding shape: ', face_embedding.shape)
        # print('audio_embedding shape: ', audio_embedding.shape)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1, )
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # print('embedding shape: ', audio_embedding.shape, face_embedding.shape)

        return audio_embedding, face_embedding


# logloss = nn.MSELoss()
logloss = nn.BCELoss()
# logloss = nn.L1Loss()


def contrastive_loss_with_mining(audio_features, img_features, temperature=0.07):
    """
    实现对比学习损失，包含在线负样本挖掘
    参数:
        audio_features: 音频特征 [batch_size, feature_dim]
        img_features: 图像特征 [batch_size, feature_dim]
        temperature: 温度参数，控制softmax的平滑度
    """
    batch_size = audio_features.size(0)
    # 特征归一化
    # audio_features = F.normalize(audio_features, p=2, dim=1)
    # img_features = F.normalize(img_features, p=2, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(audio_features, img_features.T) / temperature
    # 构建标签：对角线为正样本对（值为1），其他为负样本对（值为0）
    labels = torch.eye(batch_size, dtype=torch.float32).cuda()

    # 计算对比学习损失
    # 对音频到图像的对比损失
    audio_to_img_loss = -torch.sum(labels * F.log_softmax(similarity_matrix, dim=1), dim=1).mean()
    # 对图像到音频的对比损失
    img_to_audio_loss = -torch.sum(labels * F.log_softmax(similarity_matrix.T, dim=1), dim=1).mean()

    # 总损失为双向损失的平均
    total_loss = (audio_to_img_loss + img_to_audio_loss) / 2

    return total_loss

# 值的大小与 batchsize 相关
def info_nce_loss(audio_features, img_features, temperature=0.1):
        # 计算相似度矩阵
        sim_matrix = torch.matmul(audio_features, img_features.T) / temperature
        
        # 创建标签：对角线上的元素应该有最高的相似度
        labels = torch.arange(sim_matrix.shape[0]).long().cuda()
        
        # 计算交叉熵损失
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.T, labels)
        
        # 总损失是两个方向的平均
        total_loss = (loss_i + loss_t) / 2
        return total_loss


def cosine_loss(a, v, y):
    d = (nn.functional.cosine_similarity(a, v) + 1) / 2
    loss = logloss(d.unsqueeze(1), y)

    return loss


def validate(model, val_data_loader):
    model.eval()  # set the model to evaluation mode
    total_loss = 0
    cosine_loss_total = 0.0
    with torch.no_grad():
        for batch in val_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            cos_loss = cosine_loss(audio_embedding, face_embedding, y)
            cosine_loss_total += cos_loss.item()
            loss = info_nce_loss(audio_embedding, face_embedding)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_data_loader)
    avg_cosine = cosine_loss_total / len(val_data_loader)
    model.train()  # set the model back to training mode

    return avg_loss, avg_cosine


def train(save_dir, dataset_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_dataset = Dataset(dataset_dir, is_train=True)
    train_data_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=16)

    val_dataset = Dataset(dataset_dir, is_train=False)
    val_data_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=4)

    model = SyncNet_color().cuda()

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)

    iter_n = 0

    for epoch in range(50):
        for batch in tqdm.tqdm(train_data_loader):
            iter_n += 1
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            # print(imgT.shape, audioT.shape)
            audio_features, img_features = model(imgT, audioT)
            loss = info_nce_loss(audio_features, img_features)
            cos_loss = cosine_loss(audio_features, img_features, y)
            loss = loss + 0.1 * cos_loss
            # loss = cos_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
            optimizer.step()

            if iter_n % 100 == 0:
                print(iter_n, 'constra_loss: ', loss.item(), 'cos_loss: ', cos_loss.item())

        # print(epoch, loss.item())
        val_loss, val_cosine = validate(model, val_data_loader)
        print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss}', f'Val Cosine: {val_cosine}')

        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, str(epoch)+'.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # img = torch.zeros([1,3,160,160])
    # # audio = torch.zeros([1,128,16,32])
    # audio = torch.zeros([1,16,32,32])
    # audio_embedding, face_embedding = syncnet(img, audio)
    # print(audio_embedding.shape, face_embedding.shape)
    train(opt.save_dir, opt.dataset_dir)

    # python syncnet.py --save_dir ../models/syncnet --dataset_dir ../data
