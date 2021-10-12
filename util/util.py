from __future__ import print_function

from random import sample

import numpy as np
import os
from models.layers.mesh_prepare import fill_from_file
from os import path
import torch
from models.losses.loss import ChamferDist
from util.sampler import sampler


def mesh_upsampler(filename, dir_in, dir_out, epoch, nface):
    file_in = "%sep%d_out.obj" % (filename.split('ep')[0], epoch)
    path_in = os.path.join(dir_in, file_in)
    path_tmp = os.path.join(dir_out, 'tmp.obj')
    path_out = os.path.join(dir_out, filename)

    # upsample
    os.system(r"util/Manifold/build/manifold %s %s" % (path_in, path_tmp))
    # simply to fix face number (epoch*nface)
    os.system(r"util/Manifold/build/simplify -i %s -o %s -f %d" % (path_tmp, path_out, int(nface * 2)))
    os.system(r"rm %s" % path_tmp)

def midpoint_upsampler(faces,vs,face_uv=None,uv=None):
    def midpoint(faces,vs,size = 3):
        vs = vs.tolist()
        #faces = faces.tolist()
        face_num = len(faces)
        new_faces = []
        dic = {}
        for  i in range(face_num):
            set_dic = {}
            p_list = []
            for j in range(3):
                p1 = faces[i][j]
                p2 = faces[i][(j+1)%3]
                tp = (p1,p2)
                if tp not in dic:
                    tmp_vs = [(vs[p1][k]+vs[p2][k])/2 for k in range(size)]
                    vs.append(tmp_vs)
                    dic[(p1,p2)] = len(vs) - 1
                    dic[(p2,p1)] = len(vs) - 1
                    p3 = len(vs) - 1
                else :
                    p3 = dic[tp]
                tmpset = set()
                tmpset.add(p1)
                tmpset.add(p2)
                set_dic[p3] = tmpset
                p_list.append(p3)
            new_faces.append(p_list)
            for j in range(3):
                p1 = p_list[j]
                p2 = p_list[(j+1)%3]
                p3 = -1
                for k in range(3):
                    if (faces[i][k] in set_dic[p1]) and (faces[i][k] in set_dic[p2]):
                        p3 = faces[i][k]
                new_faces.append([p2,p1,p3])
        return np.array(new_faces),np.array(vs)

    faces,vs = midpoint(faces,vs)
    if uv is not None:
        face_uv,uv = midpoint(face_uv,uv,size=2)
        return faces,vs,face_uv,uv
    else:
        return faces,vs





def evaluate_chamfer(vs = None, vs_gt = None, path1 = None,path2 = None,sample = False ,sample_num = 100000):
    chamfer_dist = ChamferDist()
    if path2!=None and path2!=None:
        vs, f, vc_gt, uv_gt, face_uv_gt, texture_gt, text_gt = fill_from_file(file=path1)
        vs_gt, f_gt, vc_gt, uv_gt, face_uv_gt, texture_gt, text_gt = fill_from_file(file=path2)
        vs = torch.from_numpy(vs).float().unsqueeze(0).cuda()
        vs_gt = torch.from_numpy(vs_gt).float().unsqueeze(0).cuda()
        f = torch.from_numpy(f).long().unsqueeze(0).cuda()
        f_gt = torch.from_numpy(f_gt).long().unsqueeze(0).cuda()
        if sample:
            sampler_coord, _,_ = sampler(f, vs,sample_num)
            sampler_gt, _,_ = sampler(f_gt,vs_gt,sample_num)
            dist1, dist2, idx1, idx2 = chamfer_dist(sampler_coord,sampler_gt)
        else:
            dist1, dist2, idx1, idx2 = chamfer_dist(vs,vs_gt)
    else:
        vs = torch.from_numpy(vs).float().unsqueeze(0).cuda()
        vs_gt = torch.from_numpy(vs_gt).float().unsqueeze(0).cuda()
        dist1, dist2, idx1, idx2 = chamfer_dist(vs,vs_gt)
    chamfer_loss = (torch.mean(dist1) + torch.mean(dist2))
    return chamfer_loss,dist1,dist2,idx1,idx2

def create_convex(path_in, dir_out):
    filename = path_in.split('/')[-1]
    path_out = os.path.join(dir_out, "%s_ep1.obj" % filename.split('.obj')[0])
    path_tmp = "%s_tmp.obj" % path_out.split('.obj')[0]
    os.system(r"xvfb-run -a -s \"-screen 0 800x600x24\" meshlabserver -i %s -o %s -s util/Manifold/build/convex.mlx" % (
        path_in, path_tmp))
    # os.system(r"meshlabserver -i %s -o %s -s util/Manifold/build/convex.mlx" % (path_in, path_tmp))

    # upsample
    os.system(r"util/Manifold/build/manifold %s %s" % (path_tmp, path_tmp))
    # simply to fix face number (epoch*nface)
    os.system(r"util/Manifold/build/simplify -i %s -o %s -f %d" % (path_tmp, path_out, 1500))
    os.system(r"rm %s" % path_tmp)


def save_obj(vs, faces, dir, filename, colors=None, uvs=None, text=None, face_uvs=None):
    with open(os.path.join(dir, filename), 'w') as f:
        if text is not None:
            f.write("%s\n" % text)
        if uvs is not None:
            for uv in uvs:
                f.write("vt %f %f\n" % (uv[0], uv[1]))
        for i, item in enumerate(vs):
            vcol = ' %f %f %f' % (colors[i][0], colors[i][1], colors[i][2]) if colors is not None else ''
            f.write("v %f %f %f%s\n" % (item[0], item[1], item[2], vcol))

        for i, item in enumerate(faces):
            if uvs is not None:
                f.write("f %d/%d %d/%d %d/%d\n" % (item[0] + 1, face_uvs[i][0] + 1, item[1] + 1,
                                                   face_uvs[i][1] + 1, item[2] + 1, face_uvs[i][2] + 1))
            else:
                f.write("f %d %d %d\n" % (item[0] + 1, item[1] + 1, item[2] + 1))
        f.close()

def edge_counter(faces):
    dic = {}
    cnt = 0
    for face in faces:
        for i in range(3):
            if (face[i],face[(i+1)%3]) not in dic:
                cnt+=1
                dic[(face[i],face[(i+1)%3])] = 1
                dic[(face[(i+1)%3],face[i])] = 1
    return cnt

def mesh_segmentation(filename, parts=-1, vs_med=None, lim=None, laxarr=None ,laxnum = 0.):
    vs, faces, vc, uvs, face_uvs, texture, text,_ = fill_from_file(file=filename)
    face_num = faces.shape[0]
    print('Mesh total faces:%d' % face_num)
    lax = laxnum
    if lim is None:
        lim = [np.max(vs[:, i]) - np.min(vs[:, i]) for i in range(3)]
    if parts == 1:
        return 1, face_num, 0, 0, vs.shape[0], faces.tolist(), None, None,uvs, face_uvs
    if parts == -1:
        if face_num < 8100:
            return 1, face_num, 0, 0, vs.shape[0], faces.tolist(), None, None,uvs, face_uvs
        elif face_num < 12100:
            parts = 2
            lax = laxarr[0]
        elif face_num < 20100:
            lax = laxarr[1]
            parts = 4
        elif face_num < 64100:
            lax = laxarr[2]
            parts = 8
        else:
            return -1, face_num, 0, 0, 0, 0, None, None,None,None

    face_num = 0
    masks = []
    if vs_med is None:
        vs_med = np.median(vs, axis=0)
    overlap = np.zeros(vs.shape[0], dtype=np.int64)
    dic2olds = []
    mask_group = [[vs[:, i] >= vs_med[i] - lax * lim[i] for i in range(3)],
                  [vs[:, i] < vs_med[i] + lax * lim[i] for i in range(3)]]
    if parts == 2:
        # 2 parts
        masks = [mask_group[0][0],
                 mask_group[1][0]]
    elif parts == 4:
        # 4 parts
        masks = [mask_group[0][0] * mask_group[0][1],
                 mask_group[0][0] * mask_group[1][1],
                 mask_group[1][0] * mask_group[0][1],
                 mask_group[1][0] * mask_group[1][1]]
    elif parts == 8:
        masks = [mask_group[0][0] * mask_group[0][1] * mask_group[0][2],
                 mask_group[0][0] * mask_group[0][1] * mask_group[1][2],
                 mask_group[0][0] * mask_group[1][1] * mask_group[0][2],
                 mask_group[0][0] * mask_group[1][1] * mask_group[1][2],
                 mask_group[1][0] * mask_group[0][1] * mask_group[0][2],
                 mask_group[1][0] * mask_group[0][1] * mask_group[1][2],
                 mask_group[1][0] * mask_group[1][1] * mask_group[0][2],
                 mask_group[1][0] * mask_group[1][1] * mask_group[1][2],
                 ]
    faces = faces.tolist()
    if face_uvs is not None:
        face_uvs = face_uvs.tolist()
    for part, mask in enumerate(masks):
        vs1 = vs[mask, :]
        vc1 = vc[mask, :] if vc is not None else None

        point_cnt = vs1.shape[0]
        dic2oldp = np.arange(0, vs.shape[0])
        dic2oldp = dic2oldp[mask]
        overlap[mask] += 1
        dic2oldp = dic2oldp.tolist()
        old2new = {k: i for i, k in enumerate(dic2oldp)}
        point_set = set()
        point_set_over = set()
        vs1 = vs1.tolist()
        if vc1 is not None:
            vc1 = vc1.tolist()
        for i in range(len(dic2oldp)):
            point_set.add(dic2oldp[i])
            point_set_over.add(dic2oldp[i])
        faces1 = []
        face_uvs1 = []
        for i, face in enumerate(faces):
            face_uv = face_uvs[i] if face_uvs is not None else None
            is_face = (face[0] in point_set) or (face[1] in point_set) or (face[2] in point_set)
            if is_face:
                new_face = []
                new_face_uv = []
                for j, p in enumerate(face):
                    f_uv = face_uv[j] if face_uvs is not None else None
                    if p not in point_set_over:
                        point_set_over.add(p)
                        # dic2oldp.append(p)
                        old2new[p] = point_cnt
                        # overlap[p]+=1
                        vs1.append(vs[p])
                        if vc is not None:
                            vc1.append(vc[p])
                        if face_uvs is not None:
                            new_face_uv.append(f_uv)
                        new_face.append(point_cnt)
                        point_cnt += 1
                    else:
                        if face_uvs is not None:
                            new_face_uv.append(f_uv)
                        new_face.append(old2new[p])
                faces1.append(new_face)
                face_uvs1.append(new_face_uv)
        dic2olds.append(dic2oldp)
        fname = "%s_part%d.obj" % (path.basename(filename).split('.')[0], part)
        face_num = max(face_num, len(faces1))
        save_obj(vs1, faces1, path.dirname(filename), fname, colors=vc1, uvs=uvs, text=text, face_uvs=face_uvs1)

    if face_uvs is not None:
        return parts, face_num, dic2olds, overlap, vs.shape[0], faces, vs_med, lim, uvs, face_uvs
    else:
        return parts, face_num, dic2olds, overlap, vs.shape[0], faces, vs_med, lim,uvs, face_uvs


def merge_segmentations(input, dic2olds, overlap, point_num, colors=None):
    if len(input) == 1:
        return input[0]
    output = np.zeros((point_num, 3))
    output_color = np.zeros((point_num, 3)) if colors is not None else None
    # for i in range(input.shape[0]):
    for i in range(len(input)):
        for j in range(len(dic2olds[i])):
            output[dic2olds[i][j]] += input[i][j]
            if output_color is not None:
                output_color[dic2olds[i][j]] += colors[i][j]

    output /= np.expand_dims(overlap, axis=-1)
    if output_color is not None:
        output_color /= np.expand_dims(overlap, axis=-1)

    return output


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


MESH_EXTENSIONS = [
    '.obj',
]


def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


def pad(input_arr, target_length, val=0, dim=1):
    if input_arr.shape[1]>=target_length:
        return input_arr
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


def seg_accuracy(predicted, ssegs, meshes):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy
