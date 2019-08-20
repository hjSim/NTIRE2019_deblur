import os, glob, sys
import numpy as np
from PIL import Image
import tensorflow as tf

class DeblurData():

	def __init__(self, args):
		self.test_dataset = args.test_dataset
		self.channels = args.channels
		
		src_test = os.path.join(self.test_dataset,'blur/*')
		self.list_test = glob.glob(src_test)

		self.list_test.sort()
		self.num_test = 0
		for i in self.list_test:
			self.num_test += len(glob.glob(os.path.join(i,'*')))

		print('Load all files list')    
		print("# test imgs : {} ".format(self.num_test))



def get_input(list_, cnt, pretrained_dataset):
	
	if pretrained_dataset == 'NTIRE':
		if cnt == 0:
			frames = [cnt, cnt, cnt+1]
		elif cnt == len(list_)-1:
			frames = [cnt-1, cnt, cnt]
		else:
			frames = [cnt-1, cnt, cnt+1]

		if len(list_) == 1:
			frames = [cnt, cnt, cnt]
			
	elif pretrained_dataset == 'GOPRO':
		frames = [cnt]
	# print(frames)
	imgs_blur = [np.array(Image.open(list_[f])) for f in frames] # Three blurry frames
	imgs_blur = np.array(imgs_blur)
	imgs_blur = np.expand_dims(imgs_blur, axis=0) # extend to batch dimension ; [1, 3, Height, Width, Channels]
	imgs_blur = imgs_blur.astype('float32')

	h, w= imgs_blur.shape[2:4]
	assert(h%4==0 and w%4==0),'Height and width should be multiple of 4'
	
	h_new, w_new= imgs_blur.shape[2:4]
	
	imgs_blur = (imgs_blur / 255.0)*2.0-1.0 # normalize to [-1,1]
	
	return imgs_blur, list_[cnt].split('/')[-1]
