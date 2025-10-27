import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import datetime
import numpy as np
from util import mkdir
from options import BaseOptions
from datasets import test_preprocess_lessMemoryNoTail_chooseOne, testset, singlebatch_test_save, multibatch_test_save
import warnings
from skimage import io
warnings.filterwarnings('ignore')
from models import ResUnet3D


def test_result(parameter_path = 'latest_model.pth'):
    opt = BaseOptions().parse()
    opt.patch_y = opt.patch_x
    opt.patch_t = opt.patch_t
    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUnet3D(f_maps = [16,32,64]).to(device)

    ckpt = torch.load(os.path.join(os.getcwd(),opt.checkpoints_dir) + '/' + opt.task_name + '/model/' + parameter_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    output_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'result')
    opt.results_save_path = output_path
    mkdir(output_path)

    # get stacks for processing
    im_folder = os.path.join(os.getcwd(),opt.datapath)
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    opt.iter_path = None
    # test all stacks
    for N in range(len(img_list)):
        for iterations in range(opt.iter_num):
            name_list, noise_img, coordinate_list, img_mean, input_data_type = test_preprocess_lessMemoryNoTail_chooseOne(opt, N, iterations)
            prev_time = time.time()
            time_start = time.time()
            denoise_img = np.zeros(noise_img.shape)
            result_file_name = img_list[N].replace('.tif', '') + '_output'+str(iterations+1)+'.tif'

            result_name = os.path.join(output_path, result_file_name)
            opt.iter_path = result_name
            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
            with torch.no_grad():
                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.to(device)

                    real_A = noise_patch
                    real_A = Variable(real_A)
                    fake_B = model(real_A)

                    batches_done = iteration
                    batches_left = 1 * len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    if iteration % 1 == 0:
                        time_end = time.time()
                        time_cost = time_end - time_start
                        print('\r[Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                            % (N + 1,len(img_list),img_list[N],iteration + 1,len(testloader),time_cost,time_left_seconds), end=' ')
                    if (iteration + 1) % len(testloader) == 0:
                        print('\n', end=' ')

                    output_image = np.squeeze(fake_B.cpu().detach().numpy())
                    raw_image = np.squeeze(real_A.cpu().detach().numpy())
                    if (output_image.ndim == 3):
                        turn = 1
                    else:
                        turn = output_image.shape[0]

                    if (turn > 1):
                        for id in range(turn):
                            o_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                single_coordinate, id, output_image, raw_image)
                            o_patch = o_patch + img_mean
                            raw_patch = raw_patch + img_mean

                            if opt.postprocess == True:
                                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = o_patch * (
                                            np.sum(raw_patch) / np.sum(o_patch)) ** 0.5
                            else:
                                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = o_patch
                    else:
                        o_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(single_coordinate, output_image, raw_image)
                        o_patch = o_patch + img_mean
                        raw_patch = raw_patch + img_mean


                        if opt.postprocess == True:
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = o_patch * (np.sum(raw_patch) / np.sum(o_patch)) ** 0.5

                        else:
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = o_patch

                del noise_img
                output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor
                del denoise_img
                output_img = np.clip(output_img, 0, 65535).astype('int32')
                # Save inference image
                if input_data_type == 'uint16':
                    output_img = np.clip(output_img, 0, 65535)
                    output_img = output_img.astype('uint16')

                elif input_data_type == 'int16':
                    output_img = np.clip(output_img, -32767, 32767)
                    output_img = output_img.astype('int16')

                else:
                    output_img = output_img.astype('int32')

                io.imsave(result_name, output_img, check_contrast=False)
                print("test result saved in:", result_name)

if __name__ == '__main__':
    test_result('latest_model.pth')


