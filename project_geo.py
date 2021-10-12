import os

import numpy as np
import torch

from models.imgsr.utils.common_utils import get_image
from models.layers.mesh_prepare import fill_from_file
from models.losses.loss import ChamferDist
from util.sampler import sampler_uv
from util.util import save_obj
import argparse
parser = argparse.ArgumentParser(description='geo proj script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('name', metavar='NAME',
                    help='names')
torch.random.manual_seed(0)
def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_texture(texture_path):
    texture = get_image(texture_path)[1]
    return torch.from_numpy(texture).unsqueeze(0).cuda()
def load_obj(path):
    obj = {}
    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file=path)
    # gt_points = (vs-np.min(vs))/(np.max(vs)-np.min(vs)+1e-10)
    # save_obj(gt_points, [], "", "x.obj")
    obj['vs'] = torch.from_numpy(vs).unsqueeze(0).cuda()

    obj['faces'] = torch.from_numpy(faces).unsqueeze(0).cuda()
    if uvs is not None:
        obj['uvs'] = torch.from_numpy(uvs).unsqueeze(0).cuda()
        obj['face_uvs'] = torch.from_numpy(face_uvs).unsqueeze(0).cuda()
        if texture is not None:
            obj['texture'] = torch.from_numpy(texture).unsqueeze(0).cuda()
    if vc is not None:
        obj['vc'] = torch.from_numpy(vc).unsqueeze(0).cuda()
    else:
        obj['vc'] = torch.ones_like(obj['vs'])

    return obj

n = 1000000
args = parser.parse_args()
ids = args.name.split(',')

for i in ids:
    gt_path = "%s.obj"%i

    gt = load_obj(gt_path)
    #gt_points, _, gt_uvs = sampler_uv(gt['faces'], gt['vs'], 25000,  uvs=gt['uvs'], face_uvs=gt['face_uvs'])
    #gt_colors = uv2color(gt_uvs[0], gt['texture'][0])
    #save_obj(vs=gt_points[0],faces=[],colors=gt_colors,dir='',filename='pc.obj')
    gt_points = gt['vs']
    mid_point = torch.mean(gt_points, dim=1, keepdim=True)
    gt_dist = torch.sqrt(torch.sum((gt_points-mid_point)**2, dim=-1, keepdim=True))
    gt_colors = gt['vc']

    mesh_path = "%s_p_uv.obj"%i#"datasets/train/dogt1_ep1_step3000.obj"
    #17.90 17.89 17.61
    mesh = load_obj(mesh_path)
    #mesh['texture'] = get_texture("datasets/result/2/2l_ep1_step1500.jpg")
    mesh_points, _, mesh_uvs = sampler_uv(mesh['faces'], mesh['vs'], n,  uvs=mesh['uvs'], face_uvs=mesh['face_uvs'])
    dist1, dist2, idx1, idx2 = ChamferDist()(gt_points.float(), mesh_points)
    pc_uv = mesh_uvs[0][idx1[0].long()].cpu().numpy()
    np.save("input/mesh%s_uv.npy"%i, pc_uv)
    pc_colors = gt_colors[0].cpu().numpy()

    pc_xyz = gt_points[0].cpu().numpy()
    np.save("input/mesh%s_pcxyz.npy"%i, pc_xyz)
    np.save("input/mesh%s_pcrgb.npy"%i, pc_colors)
    # pc_dist = gt_dist[0].cpu().numpy()
    # pc_xyz_n = (pc_xyz-np.min(pc_xyz, axis=0, keepdims=True))/(np.max(pc_xyz,
    #                             axis=0, keepdims=True)-np.min(pc_xyz, axis=0, keepdims=True)+1e-10)
    # save_obj(pc_xyz, [], "/root/data/mesh0/", "x.obj", colors=pc_xyz_n)
    # pc_dist_n = pc_dist#(pc_dist-np.min(pc_dist, axis=0, keepdims=True))/(np.max(pc_dist,
    #                            # axis=0, keepdims=True)-np.min(pc_dist, axis=0, keepdims=True)+1e-10)
    #
    # img_np = np.ones((1024 * 1024, 3))
    # img_np_xyz = np.ones((1024 * 1024, 3))
    # img_np_dist = np.zeros((1024 * 1024, 1))
    # mask = np.zeros((1024 * 1024, 1))
    # index = np.ones_like(pc_uv[:, 1])
    # index[:] = np.floor((1-pc_uv[:,1])*1024).astype(np.uint32) * 1024 + np.around(pc_uv[:,0] * 1024).astype(np.uint32)
    # index = index.astype(np.long)
    # img_np[index] = pc_colors
    # img_np_xyz[index] = pc_xyz_n
    # img_np_dist[index] = pc_dist_n
    # mask[index] = 1
    # # index[:] = (np.clip((1-pc_uv[:,1])*1024+1,0,1023)).astype(np.uint32) * 1024 + (pc_uv[:,0] * 1024).astype(np.uint32)
    # #(np.mean(img_np, axis=1)>0).astype(np.float32)
    # # index = index.astype(np.long)
    # # img_np[index] = pc_colors
    # # mask[index] = 1
    # # index[:] = ((1-pc_uv[:,1])*1024).astype(np.uint32) * 1024 + np.clip((pc_uv[:,0] * 1024+1), 0, 1023).astype(np.uint32)
    # #
    # # index = index.astype(np.long)
    # # img_np[index] = pc_colors
    # # mask[index] = 1
    # # index[:] = (np.clip((1-pc_uv[:,1])*1024+1,0,1023)).astype(np.uint32) * 1024 + \
    # #            np.clip((pc_uv[:,0] * 1024+1), 0, 1023).astype(np.uint32)
    # #
    # # index = index.astype(np.long)
    # # img_np[index] = pc_colors
    # # mask[index] = 1
    #
    # img_np.resize((1024, 1024, 3))
    # img_np_xyz.resize((1024, 1024, 3))
    # img_np_dist.resize((1024, 1024))
    # ar = np.clip(img_np*255,0,255).astype(np.uint8)
    #
    # ar_xyz = np.clip(img_np_xyz*255,0,255).astype(np.uint8)
    # ar_dist = np.clip(img_np_dist*255,0,255).astype(np.uint8)
    #
    # mask.resize((1024, 1024))
    # mask_ar = np.clip(mask*255,0,255).astype(np.uint8)
    #
    # Image.fromarray(ar, mode="RGB").save("input/texture2.jpg")
    # Image.fromarray(ar_xyz, mode="RGB").save("input/xyz2.jpg")
    # Image.fromarray(ar_dist).save("input/dist2.jpg")
    # Image.fromarray(mask_ar).save("input/mask2.jpg")
    # np.save("input/mask2.npy", mask.astype(np.uint8))
    # np.save("input/texture2.npy", img_np)
    # np.save("input/xyz2.npy", img_np_xyz)
    # np.save("input/dist2.npy", img_np_dist)
    print("done")
    #mesh_vc = uv2color(obj1_uvs[0], mesh['texture'][0]).unsqueeze(0)


    # dist1, dist2, idx1, idx2 = ChamferDist()(obj1_points, mesh_points)


    # chamfer1 = torch.mean(torch.abs(mesh_vc[0][idx1[0].long()]-obj1_vc))
    # chamfer2 = torch.mean(torch.abs(obj1_vc[0][idx2[0].long()]-mesh_vc))

