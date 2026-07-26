# STRIDE
A generally applicable self-supervised denoiser for optical microscopy.

## 📄 Paper:
The paper is under review.


## 📘 Introduction:
This repository contains all Python code for denoising tasks. An ImageJ/Fiji plugin is provided for convenience.
![image](https://github.com/ZhangJD-ong/STRIDE/blob/main/Img/Framework1.png)


## 💻 Setup

### Installation
Clone the repository and install required packages:
```
conda env create -f enviroment.yml
conda env activate Denoising
pip install -r requirements.txt
```
### Step 1: Prepare Your Dataset
* Place your noisy 3D image stack in the ./python/data folder (at least 16 z/t frames are required)
* We've provided a sample noisy image in the ./python/data folder (Image size: 40(t) × 512(y) × 512(x))
  
### Step 2: Configure Denoising Parameters
Edit options.py to set appropriate parameters:
* Update datapath to point to your data location
* Set image_type to xyz for structural images or xyt for functional images
* Specify task_name to organize trained models and results
  
### Step 3: Train and Inference
Run the following command to train the model and perform inference with default settings:
```
cd python
python train.py
```
Or you can change settings via:
```
cd python
python train.py \
  --datapath ./data \
  --task_name Demo_Calcium \
  --image_type xyt \
  --epoch 20 \
  --iter_num 3
```
Results will be saved at ./checkpoints/task_name/result

### Step 4: Inference with Pre-trained Model (Optional)
* Replace the old data with new data in the ./python/data folder
* Run
```
python test.py
```
### Step 5: Select the optimal result (Optional)
Select the optimal results with the highest score by running:
```
python Inference_guidance.py
```
We highly recommend selecting the optimal results by considering both the score and visual assessment.

![image](https://github.com/ZhangJD-ong/STRIDE/blob/main/Img/Results.png)

## 🛠️ ImageJ/Fiji Plugin: STRIDE

### Install Required Python Packages
The plugin requires a PyTorch environment with all necessary dependencies, following the same installation procedure described above.
Attention: Install the packages in the base enviroment!!!
### Install ImageJ/Fiji
Download ImageJ/Fiji from the [official website](https://imagej.net/software/fiji/downloads)

### Deploy the STRIDE Plugin
*  Download the STRIDE plugin package from the Plugin folder in this repository
*  Copy the STRIDE folder to your ImageJ/Fiji plugins directory (./fiji-win64/Fiji.app/plugins)
*  Restart ImageJ/Fiji

### Verify Installation
* Check the plugins menu for the STRIDE option
* Open a 3D image stack in ImageJ and launch the STRIDE plugin
* Set epoch = 1, iter_num = 1, t = 16 and start denoising
* If no results appear within 1 hour, the installation may have failed


### Using STRIDE
Once successfully installed:
* Open a single noisy image in ImageJ
* Click STRIDE in the plugins menu and configure parameters (default settings are available)
* Processing typically takes 1-5 hours, depending on data size and GPU capability
* Denoised images will display automatically - remember to save them before closing


## 📚 Acknowledgements

I would like to thank the following developers and projects whose work I have used or been inspired by:

- [SRDTrans](https://github.com/cabooster/SRDTrans) – Used for dataloader process.
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) – Used for whole denosing framework.


Thanks to all contributors of the above projects for their excellent work!






