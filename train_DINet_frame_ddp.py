import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'   # 2,3是GPU的序号，根据需要进行修改。

from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from torch.utils.data import DataLoader
from dataset.dataset_DINet_frame import DINetDataset
# from sync_batchnorm import convert_model
from config.config import DINetTrainingOptions

import time
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from syncnet import SyncNet_color
from face_alignment import FaceAlignment, LandmarksType
import math
from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn import SyncBatchNorm



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
        frame training code of DINet
        we use coarse-to-fine training strategy
        so you can use this code to train the model in arbitrary resolution
    '''
    # load config
    train_opt = DINetTrainingOptions()
    # for ddp
    train_opt.parser.add_argument("--local-rank", default=-1, type=int)
    opt = train_opt.parse_args()
    local_rank = opt.local_rank
    # 根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl')
    
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # init network
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda().to(device)
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda().to(device)
    net_vgg = Vgg19().cuda().to(device)
    
    # 将所有的BN层转换为SyncBN层
    net_g = SyncBatchNorm.convert_sync_batchnorm(net_g).to(device)
    net_dI = SyncBatchNorm.convert_sync_batchnorm(net_dI).to(device)
    net_vgg = SyncBatchNorm.convert_sync_batchnorm(net_vgg).to(device)

    # 使用DistributedDataParallel包装模型
    net_g = DistributedDataParallel(net_g).to(device)
    net_dI = DistributedDataParallel(net_dI).to(device)
    # net_vgg = DistributedDataParallel(net_vgg).to(device) # 没有梯度

    # load training data in memory    
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    train_sampler = DistributedSampler(train_data)
    training_data_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                  num_workers=24, pin_memory=True, sampler=train_sampler, prefetch_factor=1)
    train_data_length = len(training_data_loader)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)

    writer = SummaryWriter(log_dir=os.path.join(opt.result_path, 'logs_DINet'))
    use_syncnet = True

    # coarse2fine
    if opt.coarse2fine:
        print('loading checkpoint for coarse2fine training: {}'.format(opt.coarse_model_path))
        checkpoint = torch.load(opt.coarse_model_path)['state_dict']
        # checkpoint = torch.load(opt.coarse_model_path)['state_dict']['net_g']
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     print('replace k: ', k)
        #     # name = k[7:]  # remove module.
        #     if 'module' not in k:
        #         name = 'module.' + k   # remove module.
        #     else:
        #         name = k

        #     new_state_dict[name] = v

        # net_g.load_state_dict(new_state_dict)
        net_g.load_state_dict(checkpoint['net_g'])
        net_dI.load_state_dict(checkpoint['net_dI'])

    # for syncnet
    if use_syncnet:
        syncnet = SyncNet_color().eval().cuda()
        syncnet.load_state_dict(torch.load(opt.pretrained_syncnet_path))
        print('Successfully loaded SyncNet from {}'.format(opt.pretrained_syncnet_path))

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # l1 loss
    recon_loss = nn.L1Loss().cuda()

    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    
    fa = FaceAlignment(LandmarksType.TWO_D, device='cuda', flip_input=False)
    
    total_iteration = 0

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        # 更新DistributedSampler的随机种子
        train_sampler.set_epoch(epoch)
        net_g.train()
        time_ = time.time()

        for iteration, data in enumerate(training_data_loader):
            # read data
            source_image_data,source_image_mask, reference_clip_data, deepspeech_feature, mouse_mask = data
            source_image_data = source_image_data.float().cuda()
            source_image_mask = source_image_mask.float().cuda()
            reference_clip_data = reference_clip_data.float().cuda()
            deepspeech_feature = deepspeech_feature.float().cuda()
            # network forward
            fake_out = net_g(source_image_mask,reference_clip_data,deepspeech_feature)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(source_image_data, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_dI.zero_grad()
            # compute fake loss
            _,pred_fake_dI = net_dI(fake_out)
            loss_dI_fake = criterionGAN(pred_fake_dI, False)
            # compute real loss
            _,pred_real_dI = net_dI(source_image_data)
            loss_dI_real = criterionGAN(pred_real_dI, True)
            # Combined DI loss
            loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            loss_dI.backward(retain_graph=True)

            # for param in net_dI.parameters():
            #     if param.grad is not None:
            #         # print('==grad apear nan in discriminator==')
            #         torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            
            # torch.nn.utils.clip_grad_norm_(net_dI.parameters(), 0.3)

            optimizer_dI.step()
            # (2) Update G network
            _, pred_fake_dI = net_dI(fake_out)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(source_image_data)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(target_tensor_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            # l1 recon_loss
            # l1_recon_loss = recon_loss(source_image_data, fake_out)

            # l1_recon_loss = torch.tensor(0.0)

            # landmark loss
            # out_frame = fake_out.permute(0,2,3,1)[0].detach().cpu().numpy()*255
            # source_frame = source_clip.permute(0,2,3,1)[0].detach().cpu().numpy()*255

            # if epoch > 5:
            if True:
                loss_lmks = torch.tensor(0.0)
                try:
                    for land_i in range(fake_out.shape[0]):
                        out_frame = fake_out.permute(0,2,3,1)[land_i]*255
                        source_frame = source_image_data.permute(0,2,3,1)[land_i]*255
                        # out_frame = fake_out.permute(0,2,3,1)[0].detach().cpu().numpy()*255
                        # source_frame = source_image_data.permute(0,2,3,1)[0].detach().cpu().numpy()*255
                        source_preds = fa.get_landmarks(source_frame)
                        out_preds = fa.get_landmarks(out_frame)
                        tensor_source_lmks = torch.tensor(source_preds[0])[48:68]
                        tensor_out_lmks = torch.tensor(out_preds[0])[48:68]
                        loss_lmks += (criterionL1(tensor_out_lmks, tensor_source_lmks) * 0.1)
                except:
                        loss_lmks += torch.tensor(0.0)

                loss_lmks = loss_lmks / fake_out.shape[0]

                if use_syncnet:
                    fake_out_clip_mouth = fake_out[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
                train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
                    y = torch.ones([deepspeech_feature.shape[0],1]).float().cuda()
                    audio_features, img_features = syncnet(fake_out_clip_mouth, deepspeech_feature.permute(0, 2, 1))
                    cos_loss = cosine_loss(audio_features, img_features, y)
                    info_loss = info_nce_loss(audio_features, img_features) * 0.25
                    # print('cos_loss: ', cos_loss.item(), 'info_loss: ', info_loss.item())
                    sync_loss = (0.1 * cos_loss + info_loss)
                    # sync_loss = info_loss
                else:
                    sync_loss = torch.tensor(0.0)

                l1_recon_loss = recon_loss(source_image_data, fake_out)

            else:
                l1_recon_loss = recon_loss(source_image_data, fake_out)
                loss_lmks = torch.tensor(0.0)
                sync_loss = torch.tensor(0.0)

            # print('loss_lmks: ', loss_lmks.item())

            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])

            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception
            # # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # combine perception loss and gan loss
            # loss_g =  loss_g_perception + loss_g_dI + l1_recon_loss
            loss_g =  loss_g_perception + loss_g_dI + loss_lmks + l1_recon_loss

            if use_syncnet:
                loss_g += sync_loss

            loss_g.backward()
            # for param in net_g.parameters():
            #     if param.grad is not None:
            #         # print('==grad apear nan in generator==')
            #         torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=0.5)

            optimizer_g.step()

            time_cost = time.time() - time_
            print(
                "===> Epoch[{}:{}]({}/{}): loss_g:{:.4f},  l1_recon_loss:{:.3f}, Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_sync: {:.4f} Loss_perception: {:.4f} Loss_lmks: {:.4f} lr_g = {:.7f} time_cost_per_step: {:.2f}s/it".format(
                    epoch, total_iteration, iteration, len(training_data_loader), float(loss_g.item()), float(l1_recon_loss.item()), float(loss_dI), float(loss_g_dI), float(sync_loss.item()),  float(loss_g_perception), float(loss_lmks.item()),optimizer_g.param_groups[0]['lr'], time_cost))

            time_ = time.time()

            total_iteration += 1

            if total_iteration % 5000 == 0:
                writer.add_scalar('Loss/G', loss_g.item(), total_iteration)
                writer.add_scalar('Loss/DI', loss_dI.item(), total_iteration)
                writer.add_scalar('Loss/GI', loss_g_dI.item(), total_iteration)
                writer.add_scalar('Loss/Perception', loss_g_perception.item(), total_iteration)
                writer.add_scalar('Loss/Lmks', loss_lmks.item(), total_iteration)
                writer.add_scalar('Loss/Sync', sync_loss.item(), total_iteration)
                writer.add_image('Real Image', source_image_data[0], total_iteration)
                writer.add_image('Real mask', source_image_mask[0], total_iteration)
                writer.add_image('Fake Image', fake_out[0], total_iteration)
                writer.add_image('Real Mouth', source_image_data[:, :, train_data.radius:train_data.radius + \
                    train_data.mouth_region_size, train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size][0], total_iteration)
                writer.add_image('Fake Mouth', fake_out[:, :, train_data.radius:train_data.radius + \
                    train_data.mouth_region_size, train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size][0], total_iteration)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)

        #checkpoint
        if epoch %  opt.checkpoint == 0 and dist.get_rank() == 0:
            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path, exist_ok=True)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.module.state_dict(), 'net_dI': net_dI.module.state_dict()},#
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict()}#
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))

    writer.close()


# python -m torch.distributed.launch --nproc_per_node 3 train_DINet_frame_d.py --augment_num=5 --mouth_region_size=256 --batch_size=8 --result_path=./asserts/training_model_weight/frame_training_ddp --coarse2fine --coarse_model_path ./asserts/training_model_weight/frame_training/netG_model_epoch_100.pth --pretrained_syncnet_path ./DINet/syncnet/12.pth
