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
pip install -r requirements.txt
```
### Step 1: Prepare your dataset
* Put the noisy 3D image stack in Data folder (at least 16 z/t is required).
* Here we prepared a noisy image of mouse brain in Data folder. Image size: 81(z) 256(y) 256(x).
  
### Step 2: Choose proper parameters for denoising in option.py
* Change the datapath where you strore your data.
* Change image_type. xyt for structural image and xyt for functional image.
* Change task_name to save well-trained models, denosing results for each task.
  
### Step 3: Jointly train and inference
* You can directly train the model and inference data by using:
```
python main.py
```
The results will be saved at ./checkpoints/task_name/results

### Step 4: Inference with well-trained model (optional)
* Sometimes you want to use well-trained model on other data, you can achieeve it by replacing the old data with new data in Data folder, and running:
```
python test.py
```
![image](https://github.com/ZhangJD-ong/STRIDE/blob/main/Img/Brain_results.png)

## ImageJ/Fiji plugin: STRIDE




## Citation
If you find the code or data useful, please consider citing the following papers:

* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001






