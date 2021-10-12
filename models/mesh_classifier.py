import os
import time

import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network ,evaluate_chamfer,pad
from models.imgsr.models import get_net
from models.imgsr.utils.sr_utils import get_noise
from models.imgsr.models.downsampler import Downsampler
import numpy as np
import copy

from .layers.mesh_prepare import fill_from_file


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.img_noise = None
        self.texture = None
        self.labels = None
        self.mesh = None
        self.ori_mesh = None
        self.soft_label = None
        self.loss = None
        self.text_loss = 0
        self.losses = None
        self.vs_gt = None
        self.vc_gt = None
        self.uv_gt = None
        self.face_uv_gt = None
        self.texture_gt = None
        self.text_gt = None
        self.f_gt = None
        self.part_n = None
        self.vc_input = None

        #
        #self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, 6, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.img_sr_net = None
        self.mse = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
        self.downsampler = Downsampler(n_planes=3, factor=4,
                                       kernel_type='lanczos2', phase=0.5, preserve_size=True).type(torch.cuda.FloatTensor)
        if opt.texture:
            self.img_sr_net = get_net(32, 'skip', 'reflection',
              skip_n33d=96,
              skip_n33u=96,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(torch.cuda.FloatTensor)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)
        mesh_paras = [p for p in self.net.parameters()]
        img_paras = [] if self.img_sr_net is None else [p for p in self.img_sr_net.parameters()]
        if self.is_train:
            self.optimizer = torch.optim.Adam(mesh_paras+img_paras, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        self.edge_features = data['edge_features']
        if self.opt.texture:
            self.texture = torch.from_numpy(self.texture_gt[0]).cuda()
            self.img_noise = get_noise(32, 'noise', (self.texture.shape[2]*4,
                                                     self.texture.shape[2]*4)).type(torch.cuda.FloatTensor).detach()
        # labels = torch.from_numpy(data['label']).long()
        # set inputs
        # noise = torch.zeros_like(input_edge_features)
        # noise.uniform_()
        # input_edge_features += 0.1*noise
        # self.labels = labels.to(self.device)
        self.mesh = np.asarray(data['mesh'])

        if self.opt.mesh_feat or self.opt.img_feat :
            path = os.path.join(self.opt.input_dir,"%s_t_ep%d.obj" % (self.opt.obj_filename.split('.obj')[0],1)) #1 --> self.opt.cur_epoch
            path1 = os.path.join(self.opt.input_dir,"%s_img_feat_ep%d.npy" % (self.opt.obj_filename.split('.obj')[0],1))
            path2 = os.path.join(self.opt.input_dir,"%s_mesh_feat_ep%d.npy" % (self.opt.obj_filename.split('.obj')[0],1))
            color_mesh_vs,_,_,_,_,_,_ = fill_from_file(file=path)
            img_feat = np.load(path1, encoding='latin1', allow_pickle=True).transpose()
            mesh_feat = np.load(path2, encoding='latin1', allow_pickle=True).transpose()

            edge_features = []
            for i,mesh in enumerate(self.mesh):
                _,_,_,idx1,_ = evaluate_chamfer(vs=mesh.vs,vs_gt=color_mesh_vs)
                idx1.squeeze_()

                tmp_img_feat = np.concatenate((img_feat[:,idx1[mesh.edges[:,0]].cpu().numpy()],img_feat[:,idx1[mesh.edges[:,1]].cpu().numpy()]),axis=0)
                tmp_mesh_feat = np.concatenate((mesh_feat[:,idx1[mesh.edges[:,0]].cpu().numpy()],mesh_feat[:,idx1[mesh.edges[:,1]].cpu().numpy()]),axis=0)
                tmp_img_feat = pad(tmp_img_feat, self.opt.ninput_edges)
                tmp_mesh_feat = pad(tmp_mesh_feat, self.opt.ninput_edges)

                if self.opt.mesh_feat and self.opt.img_feat:
                    extra_feature = np.concatenate((tmp_mesh_feat,tmp_img_feat),axis=0)
                elif self.opt.mesh_feat:
                    extra_feature = tmp_mesh_feat
                else:
                    extra_feature = tmp_img_feat
                edge_feature = np.concatenate((self.edge_features[i],extra_feature),axis=0)
                edge_features.append(edge_feature)
            edge_features = np.asarray(edge_features)
            self.edge_features = edge_features

        # for mesh in self.mesh:
        #     noise = np.random.randn(mesh.vs.shape[0],mesh.vs.shape[1])
        #     mesh.vs += noise*0.
        self.edge_features = torch.from_numpy(self.edge_features).float()
        self.edge_features = self.edge_features.to(self.device).requires_grad_(self.is_train)
        self.ori_mesh = copy.deepcopy(self.mesh)
        self.criterion.meshes = self.ori_mesh
        self.criterion.device = self.device

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss = 0
        self.losses = None
        torch.autograd.set_detect_anomaly(True)
    #   self.reset_losses()
        texture_out = out_LR = None
        for i in range(self.opt.batch_size):

            if self.opt.texture:
                rand_img_noise = self.img_noise.detach().clone()
                texture_out = self.img_sr_net(self.img_noise + rand_img_noise.normal_() * 0.02)
                out_LR = self.downsampler(texture_out)
                text_loss = self.mse(out_LR[0], self.texture)
                self.text_loss = text_loss
                text_loss.backward(retain_graph=True)
            self.part_n = self.criterion.part_n = i

            rand_noise = self.edge_features[self.part_n].unsqueeze(0).detach().clone()
            rand_noise = rand_noise.normal_()*self.opt.noise_factor
            rand_noise[6:,:] = 1.
            out = self.net(self.edge_features[self.part_n].unsqueeze(0) + rand_noise , np.asarray([self.mesh[self.part_n]])) #big bug...

            # out = self.net(self.edge_features[self.part_n].unsqueeze(0), np.asarray([self.mesh[self.part_n]]))
            if self.opt.color:
                loss,losses = self.criterion(out, self.vs_gt, self.f_gt, self.vc_gt, texture_out, out_LR, self.texture,
                                             self.uv_gt, self.face_uv_gt) #change
            else:
                loss,losses = self.criterion(out, [self.vs_gt,np.array(self.f_gt[i]),self.vc_gt]) #change

            loss.backward()
            self.mesh[self.part_n] = copy.deepcopy(self.ori_mesh[self.part_n])
            self.loss += loss
            self.update_losses(losses)
        self.avg_losses()
        self.optimizer.step()

    # def reset_losses(self):
    #     self.losses = {
    #         "loss": 0.,
    #         "loss_chamfer": 0.,
    #         "loss_normal": 0.,
    #         "loss_move": 0.,
    #         "loss_edge": 0.,
    #         "loss_area": 0.,
    #         "loss_lap": 0.,
    #         "loss_beam": 0.
    #     }

    def avg_losses(self):
        self.loss /= self.opt.batch_size
        for k in self.losses.keys():
            self.losses[k] /= self.opt.batch_size

    def update_losses(self,losses):
        if self.losses is None:
            self.losses = losses
            return
        for k in self.losses.keys():
            self.losses[k] += losses[k]

    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step(self.loss)
        # lr = self.optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
