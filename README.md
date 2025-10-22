# STRIDE
A universal self-supervised denoiser for fluorescence optical microscopy imaging.

## Paper:
The paper is under reviewed, arxiv preprint is availabled at: 


## Introduction:
This repository includes all python codes for denosing task. An ImageJ/Fiji plugin is provided for convience.


## Setup

### Installation
Clone and repo and install required packages:
```
git clone git@github.com:ZhangJD-ong/STRIDE.git
pip install -r requirement.txt
```
### Step 1: Prepare your dataset
* Put the noisy 3D image stack in Data folder (at least 16 z/t is required).
* Here we prepared a noisy image of mouse brain in Data folder. Image size: 100*256*256.
  
### Step 2: Choose proper parameters for denoising
* In option.py, change the datapath where you strore your data.
* In option.py change image_type. xyt for structural image and xyt for functional image.
* In option.py, change task_name to save well-trained models, denosing results for each task.
  
### Step 3: Jointly train and inference
* You can directly train the model and inference data by using:
```
python main.py
```
The results will be saved at ./checkpoints/results




### Training and testing
* For training the segmentation model, please add data path and adjust model parameters in the file: ./Train-and-test-code/options/BasicOptions.py. 
```
cd ./Train-and-test-code
python train.py
python test.py
```
### Inference on own data
* Please put the new data in the fold: ./Inference-code/Data/Original_data. The segmentation results can be find in ./Inference-code/Results/Tumor/.
```
cd ./Inference-code
python test.py
```
* We release the well-trained model (Can be downloaded from https://drive.google.com/drive/folders/1Sos8NK4zzkT1L96saffsg4EpUyjwRSjm?usp=sharing , due to the memory limitation in Github) and five samples to guide usage. Please put the download 'Trained_model' folder in ./Inference-code/.
* The data can only be used for academic research usage.
* More data are available at https://doi.org/10.5281/zenodo.8068383.

## Citation
If you find the code or data useful, please consider citing the following papers:

* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001






