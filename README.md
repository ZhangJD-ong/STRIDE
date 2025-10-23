# STRIDE
A universal self-supervised denoiser for fluorescence optical microscopy imaging.

## 📄 Paper:
The paper is under review. An arXiv preprint is available at:



## 📘 Introduction:
This repository contains all Python code for denoising tasks. An ImageJ/Fiji plugin is provided for convenience.
![image](https://github.com/ZhangJD-ong/STRIDE/blob/main/Img/Framework.png)


## 💻 Setup

### Installation
Clone the repository and install required packages:
```
git clone git@github.com:ZhangJD-ong/STRIDE.git
cd STRIDE
pip install -r requirements.txt
```
### Step 1: Prepare Your Dataset
* Place your noisy 3D image stack in the Data folder (at least 16 z/t frames are required)
* We've provided a sample noisy image of a mouse brain in the Data folder (Image size: 81(z) × 256(y) × 256(x))
  
### Step 2: Configure Denoising Parameters
Edit option.py to set appropriate parameters:
* Update datapath to point to your data location
* Set image_type to xyzt for structural images or xyt for functional images
* Specify task_name to organize trained models and results
  
### Step 3: Train and Inference
Run the following command to train the model and perform inference:
```
python main.py
```
Results will be saved at ./checkpoints/task_name/results

### Step 4: Inference with Pre-trained Model (Optional)
* Replace the old data with new data in the Data folder
* Run
```
python test.py
```
![image](https://github.com/ZhangJD-ong/STRIDE/blob/main/Img/Brain_results.png)

## 🛠️ ImageJ/Fiji Plugin: STRIDE

### Install Required Python Packages
The plugin requires a PyTorch environment with all necessary packages:
```
pip install -r requirements.txt
```
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

## 📖 Citation
If you find the code or data useful, please consider citing the following paper:
Zhang et al., 




