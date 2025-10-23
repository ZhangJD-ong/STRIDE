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

## 🛠️ ImageJ/Fiji plugin: STRIDE

### Install required python packages:
This plugin still works based on Pytorch enviroment, all required packages need to be installed first.
```
pip install -r requirements.txt
```
### Install ImageJ/Fiji:
You can download the ImageJ/Fiji via [ImageJ/Fiji](https://imagej.net/software/fiji/downloads)

### Deploy denoising plugin (STRIDE)
*  Download the STRIDE plugin package from this repository (Plugin folder).
*  Find the location where you install your ImageJ/Fiji, put the STRIDE folder in plugins folder. (./fiji-win64/Fiji.app/plugins)
*  Restart Imagej/Fiji

### Test if you install the STRIDE sucessfully
* In the plugins menu, see if you have STRIDE option.
* Open a 3D image stack in ImageJ, and click the STRIDE plugin.
* Choose epoch = 1, iter_num = 1, t = 16 and start denosing. If you don't get any results within 1 hour, you may fail to install STRIDE.

### STRIDE usage:
Once the plugin is successfully installed, congratulations — you're ready to use it! Just follow the steps below:
* Open the noisy image in ImageJ. Remenber only one image everytime.
* Click STRIDE in plugins menu, choose proper paramters. You can also use the default settings.
* Wait about 1-5 hours, depending on your data size and GPUs.
* All denoised images will be displed in the screen after denosing. Remenber save them before closing them.


## 📚 Acknowledgements

I would like to thank the following developers and projects whose work I have used or been inspired by:

- [SRDTrans](https://github.com/cabooster/SRDTrans) – Used for dataloader process.
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) – Used for whole denosing framework.


Thanks to all contributors of the above projects for their excellent work!

## 📖 Citation
If you find the code or data useful, please consider citing the following papers:

* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001






