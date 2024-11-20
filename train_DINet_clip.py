import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'   # GPU的序号，根据需要进行修改。

from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from models.Syncnet import SyncNetPerception
from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip import DINetDataset

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from syncnet import SyncNet_color
from face_alignment import FaceAlignment, LandmarksType
import math
from torch.utils.tensorboard import SummaryWriter

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = (nn.functional.cosine_similarity(a, v) + 1) / 2
    loss = logloss(d.unsqueeze(1), y)

    return loss


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
        # batch_size = audio_features.shape[0]
        # total_loss = total_loss * batch_size / math.log(batch_size)

        return total_loss


if __name__ == "__main__":
    '''
    clip training code of DINet
    in the resolution you want, using clip training code after frame training
    '''
    # load config
    opt = DINetTrainingOptions().parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    # training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True, num_workers=24, pin_memory=True, prefetch_factor=1)
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_dV = Discriminator(opt.source_channel * 4, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    # net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_g.to(devices)
    net_dI = nn.DataParallel(net_dI)
    net_dI.to(devices)
    net_dV = nn.DataParallel(net_dV)
    net_dV.to(devices)
    net_vgg = nn.DataParallel(net_vgg)
    net_vgg.to(devices)
    #setup optimizer
    opt.lr_g = opt.lr_g
    opt.lr_dI = opt.lr_g

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    writer = SummaryWriter(log_dir=os.path.join(opt.result_path, 'logs_DINet'))
    optimizer_dV = optim.Adam(net_dV.parameters(), lr=opt.lr_dI)
    ## load frame trained DInet weight
    print('loading frame trained DINet weight from: {}'.format(opt.pretrained_frame_DINet_path))
    checkpoint = torch.load(opt.pretrained_frame_DINet_path)
    net_g.load_state_dict(checkpoint['state_dict']['net_g'])
    net_dI.load_state_dict(checkpoint['state_dict']['net_dI'])
    print('loading frame trained DINet weight successfully!')
    
    use_syncnet = True

    # for syncnet
    if use_syncnet:
        syncnet = SyncNet_color().eval().cuda()
        syncnet.load_state_dict(torch.load(opt.pretrained_syncnet_path))
        print('Successfully loaded SyncNet from {}'.format(opt.pretrained_syncnet_path))

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    recon_loss = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).cuda()

    fa = FaceAlignment(LandmarksType.TWO_D, device='cuda', flip_input=False)

    total_iteration = 0

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_g.train()
        time_ = time.time()
        for iteration, data in enumerate(training_data_loader):
            # forward
            # source_clip, source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full = data  # source_clip -> [B, 5, 3, W, H]
            source_clip, source_clip_mask, reference_clip,deep_speech_clip = data  # source_clip -> [B, 5, 3, W, H]
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()    # [B, 5, 3, W, H] -> 5 * [B, 1, 3, W, H] -> [5*B, 3， W, H]
            source_clip_mask = torch.cat(torch.split(source_clip_mask, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            # deep_speech_full = deep_speech_full.float().cuda()
            # print(source_clip.shape, reference_clip.shape, deep_speech_clip.shape)
            fake_out = net_g(source_clip_mask,reference_clip,deep_speech_clip)
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')
            # (1) Update DI network
            optimizer_dI.zero_grad()
            _,pred_fake_dI = net_dI(fake_out)
            loss_dI_fake = criterionGAN(pred_fake_dI, False)
            _,pred_real_dI = net_dI(source_clip)
            loss_dI_real = criterionGAN(pred_real_dI, True)
            # Combined DI loss
            loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            loss_dI.backward(retain_graph=True)
            optimizer_dI.step()

            # (2) Update DV network
            optimizer_dV.zero_grad()
            condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            _, pred_fake_dV = net_dV(condition_fake_dV)
            loss_dV_fake = criterionGAN(pred_fake_dV, False)
            condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
            _, pred_real_dV = net_dV(condition_real_dV)
            loss_dV_real = criterionGAN(pred_real_dV, True)
            # Combined DV loss
            loss_dV = (loss_dV_fake + loss_dV_real) * 0.5
            loss_dV.backward(retain_graph=True)
            optimizer_dV.step()

            # (2) Update DINet
            _, pred_fake_dI = net_dI(fake_out)
            _, pred_fake_dV = net_dV(condition_fake_dV)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(source_clip)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(source_clip_half)
            perception_fake_half = net_vgg(fake_out_half)

            # landmark loss
            # out_frame = fake_out.permute(0,2,3,1)[0].detach().cpu().numpy()*255
            # source_frame = source_clip.permute(0,2,3,1)[0].detach().cpu().numpy()*255
            out_frame = fake_out.permute(0,2,3,1)[0]*255
            source_frame = source_clip.permute(0,2,3,1)[0]*255

            # if epoch > 5:
            if True:
                # try:
                #     source_preds = fa.get_landmarks(source_frame)
                #     out_preds = fa.get_landmarks(out_frame)
                #     tensor_source_lmks = torch.tensor(source_preds[0])
                #     tensor_out_lmks = torch.tensor(out_preds[0])
                #     loss_lmks = criterionL1(tensor_out_lmks, tensor_source_lmks) * 0.1
                # except:
                #     loss_lmks = torch.tensor(0.0)
                loss_lmks = torch.tensor(0.0)
                try:
                    for land_i in range(fake_out.shape[0]):
                        out_frame = fake_out.permute(0,2,3,1)[land_i]*255
                        source_frame = source_clip.permute(0,2,3,1)[land_i]*255    
                        source_preds = fa.get_landmarks(source_frame)
                        out_preds = fa.get_landmarks(out_frame)
                        tensor_source_lmks = torch.tensor(source_preds[0])
                        tensor_out_lmks = torch.tensor(out_preds[0])
                        loss_lmks += (criterionL1(tensor_out_lmks, tensor_source_lmks) * 0.2)
                except:
                        loss_lmks += torch.tensor(0.0)

                loss_lmks = loss_lmks / fake_out.shape[0]
                

                if use_syncnet:
                    fake_out_clip_mouth = fake_out[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
                train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
                    y = torch.ones([deep_speech_clip.shape[0],1]).float().cuda()
                    audio_features, img_features = syncnet(fake_out_clip_mouth, deep_speech_clip.permute(0, 2, 1))
                    cos_loss = cosine_loss(audio_features, img_features, y)
                    info_loss = info_nce_loss(audio_features, img_features)
                    # print('cos_loss: ', cos_loss.item(), 'info_loss: ', info_loss.item())
                    # sync_loss = 0.1 * cos_loss + info_loss
                    sync_loss = info_loss * 0.2
                else:
                    sync_loss = torch.tensor(0.0)

            else:
                loss_lmks = torch.tensor(0.0)
                sync_loss = torch.tensor(0.0)

            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])

            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception
            # # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # # gan dV loss
            loss_g_dV = criterionGAN(pred_fake_dV, True)
            ## sync perception loss
            # fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            # fake_out_clip_mouth = fake_out_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            # train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
            # sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
            # loss_sync = criterionMSE(sync_score, real_tensor.expand_as(sync_score)) * opt.lamb_syncnet_perception
            # combine all losses
            # torch.nan_to_num(loss_g_perception, nan=0, posinf=1e8, neginf=-1e8)
            l1_recon_loss = recon_loss(source_clip, fake_out)
            loss_g = loss_g_perception + loss_g_dI + loss_lmks + loss_g_dV + l1_recon_loss

            # l1_recon_loss = torch.tensor(0.0)

            if use_syncnet:
                loss_g = loss_g + sync_loss

            loss_g.backward()

            # for param in net_g.parameters():
            #     if param.grad is not None:
            #         # torch.nan_to_num(param.grad.data, nan=0, posinf=0, neginf=0)
            #         torch.nan_to_num(param.grad.data, nan=0, posinf=0, neginf=0, out=param.grad.data)
            #         param.grad[torch.isinf(param.grad)] = 0
            #         param.grad[torch.isnan(param.grad)] = 0
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 0.5)
            optimizer_g.step()

            time_cost = time.time() - time_
            print(
                "===> Epoch[{}:{}]({}/{}): loss_g:{:.4f},  l1_recon_loss:{:.3f}, Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_DV: {:.4f} Loss_GV: {:.4f} Loss_sync: {:.4f} Loss_perception: {:.4f} Loss_lmks: {:.4f} lr_g = {:.7f} time_cost_per_step: {:.2f}s/it".format(
                    epoch, total_iteration, iteration, len(training_data_loader), float(loss_g.item()), float(l1_recon_loss.item()), float(loss_dI), float(loss_g_dI), float(loss_dV), float(loss_g_dV), float(sync_loss.item()),  float(loss_g_perception), float(loss_lmks.item()),optimizer_g.param_groups[0]['lr'], time_cost))

                # "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_DV: {:.4f} Loss_GV: {:.4f} Loss_perception: {:.4f} lr_g = {:.7f} time_cost_per_step: {:.2f}s".format(
                #     epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),float(loss_dV), float(loss_g_dV), float(loss_g_perception),
                #     optimizer_g.param_groups[0]['lr'], time_cost))

            time_ = time.time()
            total_iteration += 1

            if total_iteration % 5000 == 0:
                writer.add_scalar('Loss/G', loss_g.item(), total_iteration)
                writer.add_scalar('Loss/DI', loss_dI.item(), total_iteration)
                writer.add_scalar('Loss/GI', loss_g_dI.item(), total_iteration)
                writer.add_scalar('Loss/Perception', loss_g_perception.item(), total_iteration)
                writer.add_scalar('Loss/Lmks', loss_lmks.item(), total_iteration)
                writer.add_scalar('Loss/Sync', sync_loss.item(), total_iteration)
                writer.add_image('Real Image', source_clip[0], total_iteration)
                writer.add_image('Real mask', source_clip_mask[0], total_iteration)
                writer.add_image('Fake Image', fake_out[0], total_iteration)
                writer.add_image('Real Mouth', source_clip[:, :, train_data.radius:train_data.radius + \
                    train_data.mouth_region_size, train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size][0], total_iteration)
                writer.add_image('Fake Mouth', fake_out[:, :, train_data.radius:train_data.radius + \
                    train_data.mouth_region_size, train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size][0], total_iteration)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)

        # checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(),'net_dI': net_dI.state_dict(),'net_dV': net_dV.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict(), 'net_dV': optimizer_dV.state_dict()}
            }

            # states = {
            #     'epoch': epoch + 1,
            #     'state_dict': {'net_g': net_g.state_dict(),'net_dI': net_dI.state_dict(),
            #     'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict()}
            # }}

            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))

# python train_DINet_clip.py --augment_num=3 --mouth_region_size=256 --batch_size=2 --pretrained_frame_DINet_path=./asserts/training_model_weight/xxxx_2.pth --result_path=./asserts/training_model_weight/xxxx --pretrained_syncnet_path ./xxxx/xxxx/12.pth
