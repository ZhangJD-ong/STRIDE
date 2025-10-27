from ij import IJ,ImagePlus
from ij.io import FileSaver,Opener
import os
import ij
from ij.gui import GenericDialog
#@ DatasetIOService ds
#@ UIService ui

imp = IJ.getImage()
fs = FileSaver(imp)
current_path = os.getcwd()
fs.saveAsTiff(os.path.join(current_path,r'plugins\STRIDE\Raw_data\Denoised.tif'))

def getOptions():
	gd = GenericDialog("Options")
	img_type = ["xyt","xyz"]
	gd.addChoice("Image type", img_type,img_type[0])
	gd.addSlider("Training epoch", 1, 50, 20)
	gd.addSlider("Iterations", 1, 10, 3)
	xy_patch = ["128", "256", "512"]
	tz_patch = ["16","32","64"]
	gd.addChoice("x and y patch size", xy_patch,xy_patch[0])
	gd.addChoice("t or z patch_size", tz_patch,tz_patch[0])
	gd.addCheckbox("2D inference mode", True)
	gd.showDialog()

	if gd.wasCanceled():
		print "User canceled dialog!"
		return

	img_type = gd.getNextChoice()
	epoch = gd.getNextNumber()
	iterations = gd.getNextNumber()
	xy_patch = gd.getNextChoice()
	tz_patch = gd.getNextChoice()
	D_center = gd.getNextBoolean()
	
	return img_type, epoch, iterations, xy_patch, tz_patch, D_center

options = getOptions()
if options is not None:
	img_type, epoch, iterations, xy_patch, t_patch, D_center = str(options[0]), int(options[1]),int(options[2]),int(options[3]),int(options[4]),options[5]
	IJ.log('Let us start training and inference! \n Please wait patiently!')
	os.system('python ./plugins/STRIDE/train.py --epoch {} --patch_x {} --patch_t {} --image_type {} --inference_mode_2D {} --iter_num {}'.format(epoch,xy_patch,t_patch,img_type,D_center,iterations))

	for i in range(iterations):
		ui.show(ds.open(os.path.join(current_path,r'plugins\STRIDE\checkpoints\ImageJ_plugins\result\Denoised_output'+str(i+1)+'.tif')))

