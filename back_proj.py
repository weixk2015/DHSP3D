import argparseimport osimport numpy as npimport torchfrom midpoint import midpoint_upsamplerfrom models.imgsr.utils.common_utils import get_imagefrom models.layers.mesh_prepare import fill_from_filefrom util.sampler import uv2colorfrom util.util import save_objdef get_texture(texture_path):    texture = get_image(texture_path)[1]    return torch.from_numpy(texture).unsqueeze(0).cuda()def load_obj(path):    obj = {}    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file=path)    obj['vs'] = torch.from_numpy(vs).unsqueeze(0).cuda()    obj['faces'] = torch.from_numpy(faces).unsqueeze(0).cuda()    if uvs is not None:        obj['uvs'] = torch.from_numpy(uvs).unsqueeze(0).cuda()        obj['face_uvs'] = torch.from_numpy(face_uvs).unsqueeze(0).cuda()        if texture is not None:            obj['texture'] = torch.from_numpy(texture).unsqueeze(0).cuda()    if vc is not None:        obj['vc'] = torch.from_numpy(vc).unsqueeze(0).cuda()    else:        obj['vc'] = torch.ones_like(obj['vs'])    return objparser = argparse.ArgumentParser(description='geo proj script',                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)parser.add_argument('name', metavar='NAME',                    help='names')args = parser.parse_args()ids = args.name.split(',')for index in ids:    # gt_path = "/root/data/mesh0_1k/pcn.obj"%index    # gt = load_obj(gt_path)        #gt_points, _, gt_uvs = sampler_uv(gt['faces'], gt['vs'], 25000,  uvs=gt['uvs'], face_uvs=gt['face_uvs'])        #gt_colors = uv2color(gt_uvs[0], gt['texture'][0])        #save_obj(vs=gt_points[0],faces=[],colors=gt_colors,dir='',filename='pc.obj')#    gt_points = gt['vs'] #   pc_xyz = gt_points[0].cpu().numpy()    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file="%s_p_uv.obj"%index)    faces, vs, face_uvs, uvs = midpoint_upsampler(faces=faces, vs=vs, face_uv=face_uvs, uv=uvs)    save_obj(vs, faces, "", "%s_m.obj"%index, colors=None, uvs=uvs, text="", face_uvs=face_uvs)    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file="/home/weixingkui/data/%s_m.obj"%index)    faces, vs, face_uvs, uvs = midpoint_upsampler(faces=faces, vs=vs, face_uv=face_uvs, uv=uvs)    save_obj(vs, faces, "", "%s_m.obj"%index, colors=None, uvs=uvs, text="", face_uvs=face_uvs)    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file="/home/weixingkui/data/%s_m.obj"%index)    vc_uv = torch.from_numpy(vc[:, :]).float()    #texture = get_image('output/2hr_xyz2.jpg'%index)[1]    texture_np = np.load('output/mesh%s_hr_xyz.npy'%index)[:3]   # faces1,vs1,face_uvs1,uvs1 = midpoint_upsampler(faces,vs,face_uvs,uvs)    vs2d = uv2color(vc_uv[:,:2],torch.from_numpy(texture_np).float())    vc_mask = (vc_uv[:,2:]==1).numpy().astype(np.int8)    #vs2dx4 = uv2color(torch.from_numpy(uvs1).float(),torch.from_numpy(texture).float())    # pc_xyz_n = (pc_xyz-np.min(pc_xyz, axis=0, keepdims=True))/(np.max(pc_xyz,    #                                 axis=0, keepdims=True)-np.min(pc_xyz, axis=0, keepdims=True)+1e-10)    #vs2d_ori = vs2d.numpy()#*(np.max(pc_xyz,axis=0, keepdims=True)-np.min(pc_xyz, axis=0,                                                       #                   keepdims=True)+1e-10) + np.min(pc_xyz, axis=0, keepdims=True)    #vs2dx4_ori = vs2dx4.numpy()*(np.max(vs,axis=0, keepdims=True)-np.min(vs, axis=0, keepdims=True)+1e-10) + np.min(vs, axis=0, keepdims=True)    vs2d_ori =  vs2d.numpy()#*vc_mask + vs*(1-vc_mask)    save_obj(vs2d_ori,faces,'output_mesh','mesh%s_xyz_h.obj'%index)    nums = [8000, 16000, 24000]    for num in nums:        os.system(r"util/Manifold/build/simplify -i %s -o %s -f %d" % ('output_mesh/mesh%s_xyz_h.obj'%index,                                                                       'output_mesh/mesh%s_xyz_%d.obj'%(index, num), num))    os.system(r"cp output_mesh/mesh%s_xyz_16000.obj datasets/result/%sn/input/%s_ep1.obj"%(index,index,index))    #save_obj(vs2dx4_ori,faces1,'output','vs2dx4.obj')    #save_obj(vs1,faces1,'output','meshx4.obj')