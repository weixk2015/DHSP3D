import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad, create_convex, mesh_upsampler
import numpy as np
from PIL import Image
from models.layers.mesh import Mesh
from models.layers.mesh_prepare import fill_from_file
from models.imgsr.utils.common_utils import get_image

class SegmentationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.segmentation = opt.segmentation
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.input_dir = self.opt.input_dir
        self.output_dir = self.opt.output_dir
        self.paths = self.make_dataset(self.input_dir, self.opt.cur_epoch, self.opt.obj_filename)
        self.vs_gt, self.f_gt, self.vc_gt,self.uv_gt,self.face_uv_gt, self.texture_gt,self.text_gt = self.get_vs_gt()
        self.size = len(self.paths)
        self.mesh = None
        self.epoch = None

        # self.get_mean_std() #change
        # # modify for network later.
        # opt.nclasses = self.nclasses
        self.ninput_channels = opt.input_nc

    def __getitem__(self, index):
        path = self.paths[0]

        if self.opt.batch_size > 1:
            path = self.paths[0]
            path = "%s_part%d.obj" % (path.split('.')[0], index)



        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        # meta['text'] = None
        # if self.opt.texture:
        #     texture_path = '%s.jpg' % self.paths[0].split('.')[0]
        #     meta['text'] = get_image(texture_path)[1]
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = edge_features
        return meta

    def __len__(self):
        return self.opt.batch_size

    def get_vs_gt(self):
        res = []
        colors = []
        faces = []
        vs_gt_path = []
        uvs = []
        textures = []
        textures_np = []
        face_uvs = []
        texts = []
        for i in range(self.opt.batch_size):
            fname = "%s_part%d.obj" % (self.opt.obj_filename.split('.')[0], i)
            vs_gt_path.append(os.path.join(self.dir, fname))
        if self.opt.batch_size == 1:
            vs_gt_path = [os.path.join(self.dir, self.opt.obj_filename)]
        for path in vs_gt_path:
            vs_gt, f_gt, vc_gt, uv_gt, face_uv_gt, texture_gt, text_gt,texture_np = fill_from_file(file=path)
            res.append(vs_gt)
            faces.append(f_gt)
            colors.append(vc_gt)
            uvs.append(uv_gt)
            face_uvs.append(face_uv_gt)
            textures.append(texture_gt)
            textures_np.append(texture_np)
            texts.append(text_gt)
        return res, faces, colors, uvs, face_uvs, textures, texts

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
            assert (os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path, epoch, obj_filename):
        meshes = []
        # assert os.path.isfile(os.path.join(path,obj_filename)), '%s is not a valid directory' % path

        tmp_path = os.path.join(path, "%s_ep%d.obj" % (obj_filename.split('.obj')[0], epoch))
        meshes.append(tmp_path)
        # for root, _, fnames in sorted(os.walk(path)):
        #     for fname in fnames:
        #         if is_mesh_file(fname):
        #             path = os.path.join(root, fname)
        #             meshes.append(path)
        #             for i in range(2,epoch+1): #change
        #                 path = os.path.join(root, "%s_ep%d.obj" % (fname.split('.obj')[0],i))
        #                 meshes.append(path)
        #             break
        #     break
        return meshes


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels
