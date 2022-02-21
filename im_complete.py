
# %%
from __future__ import print_function


import argparse
import os
import argparse
parser = argparse.ArgumentParser(description='geo proj script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

from util.sampler import uv2color

import numpy as np
from models.imgsr.models import *

from models.imgsr.utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
channel_num = 3
args = parser.parse_args()
ids = args.name.split(',')

for index in ids:
	imsize = -1
	factor = 1  # 8
	enforse_div32 = 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
	PLOT = False

	pc_xyz = np.load("input/mesh%s_pcxyz.npy"%index)
	pc_uv = np.load("input/mesh%s_uv.npy"%index)

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
		num_iter = 3000
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


	pc_uv_var = np_to_torch(pc_uv).type(dtype)
	pc_xyz_var = np_to_torch(pc_xyz).type(dtype)
	pc_max = torch.mean(torch.abs(pc_xyz_var))

	# %%
	def closure():
		global i, net_input

		if reg_noise_std > 0:
			net_input = net_input_saved + (noise.normal_() * reg_noise_std)

		out_HR = net(net_input)
		mse_loss = nn.MSELoss()
		xyzs = uv2color(pc_uv_var[0], out_HR[0])
		loss = mse_loss(xyzs, pc_xyz_var[0]/pc_max)
		#loss = (loss * mask_var.float()).sum()  # gives \sigma_euclidean over unmasked elements

		#non_zero_elements = mask_var.sum()
		mse_loss_val = loss #/ non_zero_elements

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
			im_xyz.save("output/%s_%dhr_xyz.jpg" % (index, i))
		if i % 200 == 0:
			out_HR_np = torch_to_np(out_HR[:,:channel_num]*pc_max)

			np.save("output/%s_%dhr_xyz.npy" % (index, i), out_HR_np)
		print('Iteration %05d loss %.5f mse %.5f ' % (i, total_loss, mse_loss_val))

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

	out_HR_np = torch_to_np(net(net_input)*pc_max)
	np.save("output/mesh%s_hr_xyz.npy"%index, out_HR_np)
	np.save("output_2d/mesh%s_hr_xyz.npy"%index, out_HR_np)
	im = np_to_pil(out_HR_np[:channel_num])
	im.save("output/mesh%s_hr_xyz.jpg"%index )

