import os
import tifffile as tiff
import numpy as np
from options import BaseOptions

def max_min_found(img):
    max_img = np.max(img)
    min_img = np.min(img)
    return max_img, min_img

def Outlier_removal(img, percentage = 0.05):
    ref_max, ref_min = max_min_found(img)
    f1 = np.where(img > (1 - percentage) * ref_max)
    img[f1] = (1 - percentage) * ref_max
    norm_img = (img - ref_min) / ((1 - percentage) * ref_max - ref_min)
    return norm_img

def calculate_snr(image, reference_image):
    signal_mean = np.mean(reference_image*reference_image)
    noise = abs(image - reference_image)
    noise_mean = np.mean(noise*noise)
    snr = 10 * np.log10(signal_mean / noise_mean)
    return snr

def SNR_metric(img, reference, group_size = 51):
    t= img.shape[0]
    snr, slice = 0, 0
    for i in range(t-group_size+1):
        if np.mean(img[i+group_size//2,:,:]) > 0 and np.mean(np.mean(reference[i:i+group_size,:,:],axis=0)) > 0:
            img_s = Outlier_removal(img[i+group_size//2,:,:], percentage = 0.05)
            ref_s = Outlier_removal(np.mean(reference[i:i+group_size,:,:],axis=0), percentage = 0.05)
            snr += calculate_snr(img_s,ref_s)
        slice += 1
    return snr/slice



def spatial_linearity_metric(img, reference):
    spatial_correlations = []
    for t in range(img.shape[0]):
        frame1 = img[t].flatten()
        frame2 = reference[t].flatten()
        # Handle cases with no variance
        if np.std(frame1) > 0 and np.std(frame2) > 0:
            corr = np.corrcoef(frame1, frame2)[0, 1]
            spatial_correlations.append(corr)
        else:
            spatial_correlations.append(0)

    avg_spatial_correlation = np.mean(spatial_correlations)
    return avg_spatial_correlation

def image_contrast(data):
    rms = np.std(data, axis=(1, 2))
    max_possible_rms = np.ptp(data) / 2  # Maximum possible standard deviation
    contrast_score = rms / (max_possible_rms + 1e-8)
    contrast_score = np.clip(contrast_score, 0, 1)
    return np.mean(contrast_score)


opt = BaseOptions().parse()
weights = [0.6, 0.2,0.2]
file = os.listdir(opt.datapath)[-1]
raw = tiff.imread(os.path.join(opt.datapath,file))
result_path = os.path.join(opt.checkpoints_dir,opt.task_name,'result')
metrics = np.zeros([opt.iter_num,3])
for iterations in range(opt.iter_num):
    result_file_name = file.replace('.tif', '') + '_output'+str(iterations+1)+'.tif'
    img = tiff.imread(os.path.join(result_path,result_file_name))
    snr = SNR_metric(img, raw, group_size = 11)
    metrics[iterations, 0] = snr
    linearity = spatial_linearity_metric(img,raw)
    metrics[iterations, 1] = linearity
    contrast = image_contrast(img)
    metrics[iterations,2] = contrast

summary_vector = np.sum(metrics, axis=0, keepdims=True)
normalized_matrix = metrics / summary_vector
scores = normalized_matrix[:,0]*weights[0] + normalized_matrix[:,1]*weights[1] + normalized_matrix[:,2]*weights[2]
for iterations in range(opt.iter_num):
    print('For the iteration', str(iterations+1), 'image score: ', scores[iterations])

