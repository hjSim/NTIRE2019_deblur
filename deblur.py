from __future__ import print_function
import os, glob
from datetime import datetime
import numpy as np
import tensorflow as tf 
from tensorflow.contrib.framework import arg_scope,get_model_variables, assign_from_checkpoint_fn
from tensorflow.contrib import layers
from utils import *


class Deblur():
	def __init__(self, args):
		self.model_name = 'Deblur'
		# self.l1_lambda = args.l1_lambda
		# self.max_epoch = args.max_epoch
		# self.batch_size = args.batch_size
		# self.lr = args.lr
		# self.reduced_lr = self.lr
		# self.patch_size = args.patch_size
		self.kernel_size = args.kernel_size
		self.channels = args.channels
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

		print('Model arguments, [{:s}]'.format((str(datetime.now())[:-7])))
		for arg in vars(args):
			print('# {} : {}'.format(arg, getattr(args, arg)))


	def res_down_block(self, x, filters, scope = 'Residual_block'):
		with tf.variable_scope(scope):
			with arg_scope([layers.conv2d, layers.conv2d_transpose], activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
				net = layers.conv2d(x, 2*filters, stride=2, kernel_size=[5,5])
				net = layers.conv2d_transpose(net, filters, stride=2, kernel_size=[4,4])
		return x + net

	def res_up_block(self, x, filters, scope = 'Residual_block'):
		with tf.variable_scope(scope):
			with arg_scope([layers.conv2d, layers.conv2d_transpose], activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
				net = layers.conv2d_transpose(x, filters/2, stride=2, kernel_size=[4,4])
				net = layers.conv2d(net, filters, stride=2, kernel_size=[5,5])
		return x + net

	def generator(self, input_tensor, is_validation = False):
		n_res = 9

		k = self.kernel_size
		with arg_scope([layers.conv2d], kernel_size=[5,5], activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
			net = layers.conv2d(input_tensor, 32, kernel_size=[5,5])
			for res in range(n_res):
				net = self.res_down_block(net,filters=32, scope = 'Residual_block0_'+str(res+1))
			shortcut1 = net
			net = layers.avg_pool2d(net, [2,2])

			net = layers.conv2d(net, 64, kernel_size=[1,1], activation_fn=None)			
			for res in range(n_res):
				net = self.res_down_block(net,filters=64, scope = 'Residual_block1_'+str(res+1))
			shortcut2 = net
			net = layers.avg_pool2d(net, [2,2])

			net = layers.conv2d(net, 128, kernel_size=[1,1], activation_fn=None)
			for res in range(n_res/2):
				net = self.res_down_block(net, filters=128, scope = 'Residual_block2_1_'+str(res+1))
			for res in range(n_res/2):
				net = self.res_up_block(net, filters=128, scope = 'Residual_block2_2_'+str(res+1))

			net = tf.image.resize_images(net, [tf.shape(net)[1]*2, tf.shape(net)[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			net = tf.concat([net,shortcut2], axis=3)
			net = layers.conv2d(net, 64, kernel_size=[1,1], activation_fn=None)
			for res in range(n_res):
				net = self.res_up_block(net,filters=64, scope = 'Residual_block3_'+str(res+1))

			cnn = layers.conv2d(net, 64)
			cnn = tf.image.resize_images(cnn, [tf.shape(cnn)[1]*2, tf.shape(cnn)[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			cnn = tf.concat([cnn,shortcut1], axis=3)
			cnn = layers.conv2d(cnn, 32, kernel_size=[1,1], activation_fn=None)
			cnn = layers.conv2d(cnn, 32)
			rgb = layers.conv2d(cnn, 3, activation_fn = None)
			w = layers.conv2d(cnn, 1, activation_fn = tf.nn.sigmoid)

			k2d = layers.conv2d(net, 64)
			k2d = tf.image.resize_images(k2d, [tf.shape(k2d)[1]*2, tf.shape(k2d)[2]*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			k2d = layers.conv2d(k2d, 32)
			k2d = layers.conv2d(k2d, k*k,activation_fn = None)

		return rgb, w, k2d


	def make_indicies(self, B, H, W):
		
		# B, H, W = self.batch_size, self.patch_size, self.patch_size
		k = self.kernel_size
		d = 1
		x,y = tf.meshgrid(tf.range(W), tf.range(H))

		x = tf.reshape(x,[1,H,W,1])
		x = tf.tile(x,(B,1,1,k*k))  # B,H,W,k*k

		y = tf.reshape(y,[1,H,W,1])
		y = tf.tile(y,(B,1,1,k*k))  # B,H,W,k*k

		lin1 = tf.reshape( tf.range(-(d*(k-1)/2),d*(k+1)/2,delta = d,dtype=tf.int32), [1,1,1,k]) # 1,1,1,k # Dilation here? delta = 2 
		lin1 = tf.tile(lin1,(1,1,1,k))  # 1,1,1,k*k gives like a b c a b c a b c ...

		lin2 = tf.reshape( tf.range(-(d*(k-1)/2),d*(k+1)/2,delta = d,dtype=tf.int32), [1,1,1,k,1]) # 1,1,1,k,1 ################# Dilation here?
		lin2 = tf.tile(lin2,(1,1,1,1,k))  # 1,1,1,1,k*k
		lin2 = tf.reshape(lin2,(1,1,1,k*k))  # 1,1,1,k*k gives like a a a b b b c c c...

		max_y = tf.cast(H+1, tf.int32) # Not H-1 due to padding of img
		max_x = tf.cast(W+1, tf.int32) # Not W-1 due to padding of img
		zero = tf.zeros([], dtype=tf.int32)

		x_lin = tf.add(x+1,lin1) # +1 is correction due to padding of img
		y_lin = tf.add(y+1,lin2) # +1 is correction due to padding of img

		x = tf.clip_by_value(x, zero, max_x) 
		y = tf.clip_by_value(y, zero, max_y) 
		x_lin = tf.clip_by_value(x_lin, zero, max_x) 
		y_lin = tf.clip_by_value(y_lin, zero, max_y) 

		x = tf.cast(x, tf.int32)
		y = tf.cast(y, tf.int32)
		x_lin = tf.cast(x_lin, tf.int32)
		y_lin = tf.cast(y_lin, tf.int32)

		batch_idx = tf.range(0, B)
		batch_idx = tf.reshape(batch_idx, (B, 1, 1, 1))
		b = tf.tile(batch_idx, (1, H, W, k*k)) # (B, H, W)

		return tf.stack([b, y_lin, x_lin], 4)

	def local_conv(self, img, kernel_2d):
		# img (B, H, W, 3)
		# kernel_2d (B, H, W, k)

		# Convolve with kernel_2d
		result = tf.pad(img,([0,0],[1,1],[1,1],[0,0]),mode='CONSTANT',constant_values=0)
		result = tf.gather_nd(result, self.indices) # This process including generating self.indices can be replaced by the function 'tf.images.extract_patches'.

		kernel_2d = tf.expand_dims(kernel_2d, axis=-1) # (B, H, W, k, 1). Because of the RGB dimension
		result = tf.multiply(result,kernel_2d) # Elementwise multiplication. Resulting (B, H, W, k, 3)
		result = tf.reduce_sum(result,axis=3) # (B, H, W, 3)
		
		return result


	def build_model(self, args):
		data = DeblurData(args)
	
		if args.phase == 'test':
			if args.pretrained_dataset == 'NTIRE' :
				self.train_blur_frames = tf.placeholder(tf.float32, (1, 3, None, None, self.channels))
				self.train_blur = self.train_blur_frames[:,1,:]
			elif  args.pretrained_dataset == 'GOPRO' :
				self.train_blur_frames = tf.placeholder(tf.float32, (1, 1, None, None, self.channels))
				self.train_blur = self.train_blur_frames[:,0,:]
			self.list_test = data.list_test
			self.indices = self.make_indicies(B = 1, H = tf.shape(self.train_blur)[1], W = tf.shape(self.train_blur)[2])
		
		with tf.variable_scope("separable_conv", reuse = tf.AUTO_REUSE):
			if args.pretrained_dataset == 'NTIRE':
				self.rgb, self.w, self.k2d = self.generator(tf.concat([self.train_blur_frames[:,0,:],self.train_blur_frames[:,1,:],self.train_blur_frames[:,2,:]],axis=3))
			elif args.pretrained_dataset == 'GOPRO' : 
				self.rgb, self.w, self.k2d = self.generator(self.train_blur)
			self.output_k2d = self.local_conv(self.train_blur, self.k2d)
			self.output = self.w*self.output_k2d + (1-self.w)*self.rgb 

		self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))


	def test(self, args, list_test):
		saver = tf.train.Saver()
		if args.pretrained_dataset == 'NTIRE':
			ckpt = tf.train.get_checkpoint_state('checkpoints_NTIRE/')
		elif args.pretrained_dataset == 'GOPRO':
			ckpt = tf.train.get_checkpoint_state('checkpoints_GOPRO/')
			
		assert( ckpt and ckpt.model_checkpoint_path ), "There is no checkpoint to restore"
		saver.restore(self.sess, ckpt.model_checkpoint_path)
		start_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
		print("!!!!!!!!!!!!!! Restored the model pretrained on {} dataset at epoch : {}".format(args.pretrained_dataset, start_epoch))

		output_path = os.path.join(args.working_directory, 'test')
		if args.ensemble:
			output_path += '_ensemble'
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		num_test_vid = len(list_test)
		
		for test_vid in list_test:
			
			vid_folder = os.path.join(output_path,test_vid.split('/')[-1])
			if not os.path.exists(vid_folder):
				os.makedirs(vid_folder)

			test_frs = glob.glob(os.path.join(test_vid,'*'))
			test_frs.sort()
			num_test_fr = len(test_frs)
			# test_fr_cnt = 9 # Start from '00000009.png', output every 10 frames (setting for NTIRE 2019)
			test_fr_cnt = 0 # General setting
			while test_fr_cnt < num_test_fr:
				# if test_fr_cnt%10 == 9: # Start from '00000009.png', output every 10 frames (setting for NTIRE 2019)
				if True: # General setting
					input_blur_frames, name = get_input(test_frs, test_fr_cnt, args.pretrained_dataset)
					_, __, h, w, ___ = input_blur_frames.shape
					if not args.ensemble:
						tic = datetime.now()
						output = self.sess.run(self.output, {self.train_blur_frames : input_blur_frames})
						toc = datetime.now()
						
					else :
						outputs = np.zeros((8, h, w, 3)) 
						tic = datetime.now()
						for i in range(8):
							rot = i % 4
							flip = i / 4
							input_blur_temp = np.rot90(input_blur_frames,rot,(2,3))
							if flip :
								input_blur_temp = np.flip(input_blur_temp,axis = 3)
							output = self.sess.run(self.output, {self.train_blur_frames : input_blur_temp})
							if flip:
								output = np.flip(output,axis = 2)
							output = np.rot90(output,(4-rot)%4,(1,2))
							outputs[i] = output
						output = np.mean(outputs, axis = 0, keepdims = True)
						toc = datetime.now()
					test_img = np.clip(output[0],-1.0,1.0)
					test_img = Image.fromarray(((test_img + 1.0) / 2.0 * 255.0).astype(np.uint8))
					test_img.save(os.path.join(vid_folder,name))
					print("Test image {} saved, run time {}".format(test_vid.split('/')[-1]+'/'+name, toc-tic))
				test_fr_cnt += 1


	def test_psnr(self, args):
		if args.ensemble:
			output_path = './data/test_ensemble/'
		else :
			output_path = './data/test/'

		output_vids = glob.glob(output_path+'*')
		output_vids.sort()
		
		psnr = 0.0
		cnt = 0
		for output_vid in output_vids:
			output_frs = glob.glob(output_vid+'/*')
			output_frs.sort()
			num_fr = len(output_frs)
			vid_psnr = 0.0
			for output_fr in output_frs:
				test_output = Image.open(output_fr)
				test_output = np.array(test_output).astype('float32')
				
				test_sharp = Image.open(os.path.join(args.test_dataset,'sharp/',output_fr.split('/')[-2],output_fr.split('/')[-1]))
				test_sharp = np.array(test_sharp).astype('float32')
				squared_error = np.square(test_output - test_sharp)
				vid_psnr += 10 * np.log10(255.0*255.0 / np.mean(squared_error))
				cnt += 1
			print('Seq. '+ output_vid +' PNSR : {:0.2f}'.format(vid_psnr/num_fr))
			psnr += vid_psnr
		print('#########################')
		print('Avg. PNSR : {:0.2f}'.format(psnr/cnt))
