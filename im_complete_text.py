
# %%
from __future__ import print_function
import matplotlib.pyplot as plt


import argparse
import os
import argparse
parser = argparse.ArgumentParser(description='geo proj script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('name', metavar='NAME',
                    help='names')

import numpy as np
from models.imgsr.models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from models.imgsr.models.downsampler import Downsampler

from models.imgsr.utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
channel_num = 3
args = parser.parse_args()
ids = args.name.split(',')

for index in ids:
	# data_dir = "/root/data/mesh%d/" % index
	imsize = -1
	factor = 1  # 8
	enforse_div32 = 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
	PLOT = False

	# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
	# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8
	path_to_image = "input_f/texture%s.jpg"%index
	path_to_mask = "input_f/mask%s.npy"%index

	# %% md

	# Load image and baselines

	# %%

	# Starts here
	mask_orig_np = np.load(path_to_mask)
	#mask_orig_np.resize((1,1024,1024))
	# Set up parameters and net
	mask_orig_np.resize((1024, 1024))
	img_orig_np = np.load("input_f/texture%s.npy"%index).transpose([2,0,1])
	img_orig_nps = img_orig_np #np.con5kenate([img_orig_np, img_orig_np_xyz, img_orig_np_xyz], axis=0)
	#mask_orig_np = (np.min(img_orig_np, axis=0) > 0).astype(np.uint8)
	#img_orig_np = new_im
	input_depth = 32
	INPUT = 'noise'
	pad = 'reflection'
	OPT_OVER = 'net'
	KERNEL_TYPE = 'lanczos2'

	LR = 0.01
	tv_weight = 0.00

	OPTIMIZER = 'adam'

	if factor == 2:
		num_iter = 2000
		reg_noise_std = 0.01
	elif factor == 8:
		num_iter = 4000
		reg_noise_std = 0.05
	elif factor == 1:
		num_iter = 2000
		reg_noise_std = 0.02
	else:
		assert False, 'We did not experiment with other factors'

	# %%

	net_input = get_noise(input_depth, INPUT, (1024, 1024)).type(dtype).detach()

	NET_TYPE = 'skip'  # UNet, ResNet
	net = get_net(input_depth, 'skip', pad,n_channels=channel_num,
	              skip_n33d=72,
	              skip_n33u=72,
	              skip_n11=4,
	              num_scales=5,
	              upsample_mode='bilinear').type(dtype)

	# Losses


	img_LR_var = np_to_torch(img_orig_nps).type(dtype)
	mask_var = np_to_torch(mask_orig_np).type(torch.cuda.LongTensor).unsqueeze(1).repeat((1,channel_num,1,1))

	#downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


	# %% md

	# Define closure and optimize

	# %%
	def closure():
		global i, net_input

		if reg_noise_std > 0:
			net_input = net_input_saved + (noise.normal_() * reg_noise_std)

		out_HR = net(net_input)
		mse_loss = nn.MSELoss(reduction='none')

		loss = mse_loss(out_HR, img_LR_var)
		loss = (loss * mask_var.float()).sum()  # gives \sigma_euclidean over unmasked elements

		non_zero_elements = mask_var.sum()
		mse_loss_val = loss / non_zero_elements

		total_loss = mse_loss_val
		if tv_weight > 0:
			total_loss += tv_weight * tv_loss(out_HR[:,:3])
			total_loss += tv_weight * tv_loss(out_HR[:,3:6])

		total_loss.backward()
		# for name, parms in net.named_parameters():
		# 	if parms.grad is None:
		# 		print('-->name:', name)
		# 	print(parms.grad)
		if i % 200 == 0:
			im_xyz = np_to_pil(torch_to_np(out_HR[:,:channel_num]))
			#im_xyz = np_to_pil(torch_to_np(out_HR[:,3:6]))
			#im_xyz = np_to_pil(torch_to_np(out_HR[:,6:]))

			#im.save("output/%d_%dhr1.jpg" % (index,i))
			im_xyz.save("output/%s_%dhr.jpg" % (index, i))
		if i % 200 == 0:
			out_HR_np = torch_to_np(out_HR[:,:channel_num])
			#im_xyz = np_to_pil(torch_to_np(out_HR[:,3:6]))
			#im_xyz = np_to_pil(torch_to_np(out_HR[:,6:]))

			#im.save("output/%d_%dhr.jpg" % (index,i))
			np.save("output/%s_%dhr.npy" % (index, i), out_HR_np)
			#im_xyz.save("output/%d_%dhr_xyz.jpg" % (index, i))
		# Log
		#psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
		#psnr_HR = 0
		print('Iteration %05d loss %.5f mse %.5f ' % (i, total_loss, mse_loss_val))

		# History
		#psnr_history.append([psnr_LR, psnr_HR])


		i += 1

		return total_loss


	# %%

	psnr_history = []
	net_input_saved = net_input.detach().clone()
	noise = net_input.detach().clone()

	i = 0
	p = get_params(OPT_OVER, net, net_input)
	optimize(OPTIMIZER, p, closure, LR, num_iter)

	# %%

	out_HR_np = torch_to_np(net(net_input))
	np.save("output/%shr.npy"%index, out_HR_np)
	#np.save("output_2d/%dhr.npy"%index, out_HR_np)
	im = np_to_pil(out_HR_np[:channel_num])
	im.save("output/%shr.jpg" % index)
	im.save("output_2d/%shr.jpg" % index)

#im_xyz = np_to_pil(out_HR_np[3:6])
	#im_xyz.save("output/%dhr_xyz1.jpg" % index)
	#im_dist = np_to_pil(out_HR_np[6:])
	#im_dist.save("output/%dhr_dist1.jpg" % index)
	# result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])
	#
	# # For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
	# plot_image_grid([imgs['HR_np'],
	#                  imgs['bicubic_np'],
	#                  out_HR_np], factor=4, nrow=1)

	# %%


