import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def sampler(faces, verts, n, bypass=False,sigma = 0.0):
	"""Sample n verts given tri-mesh.

	:Parameters:
	  faces : batch_size * face_num * 3
	    The faces of the mesh, start from 1.
	  verts : batch_size * vert_num * 3
	    The verts of the mesh.
	  n : an int number
	     sample num
	:Return:
	  The sampled n verts: batch_size * n * 3
	"""
	if bypass:
		return verts, verts
	batch_size = faces.shape[0]
	faces_flatten = faces.view(batch_size, -1)  # b * (nv * 3)
	face_verts = verts[:, faces_flatten[0]].view(batch_size, -1, 3, 3)  # b * nf * 3 * 3
	for i in range(batch_size):
		# could batch?
		face_verts[i] = verts[i, faces_flatten[i]].view(-1, 3, 3)  # nf * 3 * 3
	v1 = face_verts[:, :, 1] - face_verts[:, :, 0]  # b * nv * 3
	v2 = face_verts[:, :, 2] - face_verts[:, :, 0]  # b * nv * 3
	# cal face areas
	areas = torch.sqrt(
		torch.abs(torch.sum(v1 * v1, dim=-1) * torch.sum(v2 * v2, dim=-1) - (torch.sum(v1 * v2, dim=-1)) ** 2)) / 2.0

	sample_verts = torch.ones(batch_size, n, 3, 3, device=faces.device)
	sample_faces = torch.ones(batch_size, n, device=faces.device).long()
	for i, area in enumerate(areas):
		# could not batch
		sample = torch.multinomial(area, n, replacement=True)  # sample weighted
		sample_faces[i] = sample
		sample_verts[i] = face_verts[i, sample]
	sample_v1 = sample_verts[:, :, 1] - sample_verts[:, :, 0]
	sample_v2 = sample_verts[:, :, 2] - sample_verts[:, :, 0]
	sample_norms = torch.cross(sample_v1, sample_v2)
	sample_norms = sample_norms / (torch.norm(sample_norms, dim=-1, keepdim=True)+1e-9)
	prob_vec1, prob_vec2 = torch.rand(batch_size, n, device=faces.device), torch.rand(batch_size,
																					  n,
																					  device=faces.device)  # uniform sample a1, a2
	mask = prob_vec1 + prob_vec2 > 1  # if a1 + a2 > 1, adjust a1 and a2
	prob_vec1[mask] = 1 - prob_vec1[mask]
	prob_vec2[mask] = 1 - prob_vec2[mask]
	target_points = sample_verts[:, :, 0] + (sample_v1 * prob_vec1.unsqueeze(-1) + sample_v2 * prob_vec2.unsqueeze(-1))
	dists = torch.min(torch.norm(sample_verts - target_points.unsqueeze(-2), dim=-1), dim=-1)[0]
	# ratio = 1/dists
	# ratio = ratio/torch.mean(ratio)
	# sigma = self.options.sample_sigma - self.options.sample_sigma * (self.options.cur_step % 500) / \
	# 		(500 * 2)
	if sigma > 0:
		ratio = torch.exp(-(dists / (2 * (sigma ** 2))))
	else:
		ratio = dists.clone()
		ratio[0,:]=1.

	return target_points, sample_norms, ratio


def sampler_color(faces, verts, n,  colors=None, bypass=False):
	"""Sample n colors given tri-mesh.

	:Parameters:
	  faces : batch_size * face_num * 3
	    The faces of the mesh, start from 1.
	  verts : batch_size * vert_num * 3
	    The verts of the mesh.
	  n : an int number
	     sample num
	:Return:
	  The sampled n verts: batch_size * n * 3
	"""
	if bypass:
		return verts, verts, colors
	if colors is None:
		return sampler(faces, verts, n)
	batch_size = faces.shape[0]
	faces_flatten = faces.view(batch_size, -1)  # b * (nv * 3)
	face_verts = verts[:, faces_flatten[0]].view(batch_size, -1, 3, 3)  # b * nf * 3 * 3
	face_colors = colors[:, faces_flatten[0]].view(batch_size, -1, 3, 3)  # b * nf * 3 * 3
	for i in range(batch_size):
		# could batch?
		face_verts[i] = verts[i, faces_flatten[i]].view(-1, 3, 3)  # nf * 3 * 3
		face_colors[i] = colors[i, faces_flatten[i]].view(-1, 3, 3)  # nf * 3 * 3
	v1 = face_verts[:, :, 1] - face_verts[:, :, 0]  # b * nv * 3
	v2 = face_verts[:, :, 2] - face_verts[:, :, 0]  # b * nv * 3
	# cal face areas
	areas = torch.sqrt(
		torch.abs(torch.sum(v1 * v1, dim=-1) * torch.sum(v2 * v2, dim=-1) - (torch.sum(v1 * v2, dim=-1)) ** 2)) / 2.0

	sample_verts = torch.ones(batch_size, n, 3, 3, device=faces.device)
	sample_colors = torch.ones(batch_size, n, 3, 3, device=faces.device)
	sample_faces = torch.ones(batch_size, n, device=faces.device).long()
	for i, area in enumerate(areas):
		# could not batch
		sample = torch.multinomial(area, n, replacement=True)  # sample weighted
		sample_faces[i] = sample
		sample_verts[i] = face_verts[i, sample]
		sample_colors[i] = face_colors[i, sample]
	sample_v1 = sample_verts[:, :, 1] - sample_verts[:, :, 0]
	sample_v2 = sample_verts[:, :, 2] - sample_verts[:, :, 0]
	sample_c1 = sample_colors[:, :, 1] - sample_colors[:, :, 0]
	sample_c2 = sample_colors[:, :, 2] - sample_colors[:, :, 0]
	prob_vec1, prob_vec2 = torch.rand(batch_size, n, device=faces.device), torch.rand(batch_size,
	                                                                                  n,
	                                                                                  device=faces.device)  # uniform sample a1, a2
	mask = prob_vec1 + prob_vec2 > 1  # if a1 + a2 > 1, adjust a1 and a2
	prob_vec1[mask] = 1 - prob_vec1[mask]
	prob_vec2[mask] = 1 - prob_vec2[mask]
	target_points = sample_verts[:, :, 0] + (sample_v1 * prob_vec1.unsqueeze(-1) + sample_v2 * prob_vec2.unsqueeze(-1))
	target_colors = sample_colors[:, :, 0] + (sample_c1 * prob_vec1.unsqueeze(-1) + sample_c2 * prob_vec2.unsqueeze(-1))
	return target_points, None, target_colors


def sampler_uv(faces, verts, n,  uvs=None, face_uvs=None,colors = None, bypass=False):
	"""Sample n uvs given tri-mesh.

	:Parameters:
	  faces : batch_size * face_num * 3
	    The faces of the mesh, start from 1.
	  verts : batch_size * vert_num * 3
	    The verts of the mesh.
	  n : an int number
	     sample num
	:Return:
	  The sampled n verts: batch_size * n * 3
	"""
	if bypass:
		return verts, verts, uvs
	if uvs is None:
		return sampler(faces, verts, n)
	batch_size = faces.shape[0]
	faces_flatten = faces.view(batch_size, -1)  # b * (nv * 3)
	face_uvs_flatten = face_uvs.view(batch_size, -1)  # b * (nv * 3)
	face_verts = verts[:, faces_flatten[0]].view(batch_size, -1, 3, 3)  # b * nf * 3 * 3
	face_uvs = uvs[:, face_uvs_flatten[0]].view(batch_size, -1, 3, 2)  # b * nf * 3 * 2
	for i in range(batch_size):
		# could batch?
		face_verts[i] = verts[i, faces_flatten[i]].view(-1, 3, 3)  # nf * 3 * 3
		face_uvs[i] = uvs[i, face_uvs_flatten[i]].view(-1, 3, 2)  # nf * 2 * 3
	v1 = face_verts[:, :, 1] - face_verts[:, :, 0]  # b * nv * 3
	v2 = face_verts[:, :, 2] - face_verts[:, :, 0]  # b * nv * 3
	# cal face areas
	areas = torch.sqrt(
		torch.abs(torch.sum(v1 * v1, dim=-1) * torch.sum(v2 * v2, dim=-1) - (torch.sum(v1 * v2, dim=-1)) ** 2)) / 2.0

	sample_verts = torch.ones(batch_size, n, 3, 3, device=faces.device)
	sample_uvs = torch.ones(batch_size, n, 3, 2, device=faces.device)
	sample_faces = torch.ones(batch_size, n, device=faces.device).long()
	for i, area in enumerate(areas):
		# could not batch
		sample = torch.multinomial(area, n, replacement=True)  # sample weighted
		sample_faces[i] = sample
		sample_verts[i] = face_verts[i, sample]
		sample_uvs[i] = face_uvs[i, sample]
	sample_v1 = sample_verts[:, :, 1] - sample_verts[:, :, 0]
	sample_v2 = sample_verts[:, :, 2] - sample_verts[:, :, 0]
	sample_u1 = sample_uvs[:, :, 1] - sample_uvs[:, :, 0]
	sample_u2 = sample_uvs[:, :, 2] - sample_uvs[:, :, 0]
	#sample_c1 = sample_colors[:, :, 1] - sample_colors[:, :, 0]
	#sample_c2 = sample_colors[:, :, 2] - sample_colors[:, :, 0]
	prob_vec1, prob_vec2 = torch.rand(batch_size, n, device=faces.device), torch.rand(batch_size,
	                                                                                  n,
	                                                                                  device=faces.device)  # uniform sample a1, a2
	mask = prob_vec1 + prob_vec2 > 1  # if a1 + a2 > 1, adjust a1 and a2
	prob_vec1[mask] = 1 - prob_vec1[mask]
	prob_vec2[mask] = 1 - prob_vec2[mask]
	target_points = sample_verts[:, :, 0] + (sample_v1 * prob_vec1.unsqueeze(-1) + sample_v2 * prob_vec2.unsqueeze(-1))
	target_uvs = sample_uvs[:, :, 0] + (sample_u1 * prob_vec1.unsqueeze(-1) + sample_u2 * prob_vec2.unsqueeze(-1))
	#target_colors = sample_colors[:, :, 0] + (sample_c1 * prob_vec1.unsqueeze(-1) + sample_c2 * prob_vec2.unsqueeze(-1))
	return target_points, None, target_uvs#target_colors

def uv2color(uvs, texture):
	new_uv = uvs.clone()
	new_uv[:, 0] = ((uvs[:, 0] - 0.5)*2)
	new_uv[:, 1] = -((uvs[:, 1] - 0.5)*2)
	colors = F.grid_sample(texture.unsqueeze(0), new_uv.unsqueeze(0).unsqueeze(0))[0, :, 0] # 3 * nf
	colors = torch.transpose(colors, 0, 1)
	return colors

def test():
	vertexes = []
	faces = []
	colors = []
	with open('mesh0.obj') as f:
		for line in f:
			if len(line.split()) == 0:
				continue

			if line.split()[0] == 'v':
				vertexes.append([float(v) for v in line.split()[1:4]])
				if len(line.split()) > 4:
					colors.append([float(v) for v in line.split()[4:7]])
			if line.split()[0] == 'f':
				faces.append([int(v) for v in line.split()[1:4]])
		faces = np.vstack(faces).astype('int32') - 1
		vertexes = np.vstack(vertexes).astype('float32')
		faces = Variable(torch.LongTensor(faces).cuda()).unsqueeze(0).repeat(2, 1, 1)
		vertexes = Variable(torch.FloatTensor(vertexes).cuda()).unsqueeze(0).repeat(2, 1, 1)
		if len(colors) > 0:
			colors = Variable(torch.FloatTensor(colors).cuda()).unsqueeze(0).repeat(2, 1, 1)
		else:
			colors = None

		points, _ = sampler_color(faces, vertexes, 100000, colors=colors)
		points = points[0].cpu().data.numpy()

		vert = np.hstack((np.full([points.shape[0], 1], 'v'), points))
		np.savetxt('res.obj', vert, fmt='%s', delimiter=' ')


if __name__ == '__main__':
	test()
