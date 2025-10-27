import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import random
import datetime
from collections import OrderedDict
import numpy as np
from util import mkdir, Train_Logger, LossAverage, group_wise_avg
from options import BaseOptions
from datasets import train_preprocess_lessMemoryMulStacks, trainset
from test import *
import warnings
warnings.filterwarnings('ignore')
from models import ResUnet3D
from sampling import *


def train (trainloader,epoch):
    global prev_time

    Loss = LossAverage()
    model.train()

    for iteration, noisy in enumerate(trainloader):
        noisy = noisy.to(device)
        mask1, mask2, mask3 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        noisy_sub3 = generate_subimages(noisy, mask3)
        if opt.patch_t == 64:
            p = random.choice([8,16,32])
        elif opt.patch_t == 32:
            p = random.choice([8,16])
        else:
            p = 8
        noisy_sub1_avg = group_wise_avg(noisy_sub1,p, opt.image_type)
        noisy_sub2_avg = group_wise_avg(noisy_sub2,p, opt.image_type)
        noisy_sub3_avg = group_wise_avg(noisy_sub3,p, opt.image_type)


        noisy_output = model(noisy_sub1)
        noisy_out_avg = group_wise_avg(noisy_output, p, opt.image_type)
        noisy_output_avg = model(noisy_sub1_avg)

        loss2neighbor_1 = 0.5*L1_pixelwise(noisy_output, noisy_sub2) + 0.5*L2_pixelwise(noisy_output, noisy_sub2)
        loss2neighbor_2 = 0.5*L1_pixelwise(noisy_output, noisy_sub3) + 0.5*L2_pixelwise(noisy_output, noisy_sub3)
        loss_self = 0.5 * loss2neighbor_1 + 0.5 * loss2neighbor_2

        loss2neighbor_1_avg = 0.5*L1_pixelwise(noisy_output_avg, noisy_sub2_avg) + 0.5*L2_pixelwise(noisy_output_avg, noisy_sub2_avg)
        loss2neighbor_2_avg = 0.5*L1_pixelwise(noisy_output_avg, noisy_sub3_avg) + 0.5*L2_pixelwise(noisy_output_avg, noisy_sub3_avg)
        loss_self_avg = 0.5 * loss2neighbor_1_avg + 0.5 * loss2neighbor_2_avg

        loss_mutual = 0.5*L1_pixelwise(noisy_output_avg,noisy_out_avg) + 0.5*L2_pixelwise(noisy_output_avg, noisy_out_avg)
        loss = loss_self + loss_self_avg + loss_mutual

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batches_done = epoch * len(trainloader) + iteration
        batches_left = opt.epoch * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        if iteration % 1 == 0:
            time_end = time.time()
            print('\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.2f] [ETA: %s] [Time cost: %.2d s] '
                % (epoch,opt.epoch,iteration + 1, len(trainloader),loss.item(), time_left,time_end - time_start), end=' ')
        if (iteration + 1) % len(trainloader) == 0:
            print('\n', end=' ')

        Loss.update(loss.item(), 1)

    return OrderedDict({'Loss': Loss.avg})


if __name__ == '__main__':
    opt = BaseOptions().parse()
    opt.patch_y = opt.patch_x
    opt.patch_t = opt.patch_t
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_name_list, train_noise_img, train_coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(opt)
    train_data = trainset(train_name_list, train_coordinate_list, train_noise_img, stack_index)
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    model = ResUnet3D(f_maps = [16,32,64]).to(device)

    param_num = sum([param.nelement() for param in model.parameters()])
    print('\033[1;31mParameters of the model is {:.2f} M. \033[0m'.format(param_num / 1e6))

    result_path = os.path.join(opt.checkpoints_dir,opt.task_name)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    model_save_path = os.path.join(result_path,'model')
    mkdir(model_save_path)
    prev_time = time.time()
    time_start = time.time()
    for epoch in range(opt.epoch):
        epoch = epoch +1
        train_log = train(trainloader,epoch)

        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))
        if epoch%opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_'+str(epoch)+'.pth'))

    test_result('latest_model.pth')
            
            
            
            
