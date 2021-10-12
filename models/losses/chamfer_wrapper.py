# torch must be imported before we import chamfer
import torch
import chamfer
import torch.nn as nn
from torch.autograd import Function


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class ChamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, _idx1, _idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDist(nn.Module):
    def __init__(self):
        super(ChamferDist, self).__init__()

    def forward(self, input1, input2):
        return ChamferFunction.apply(input1, input2)

if __name__ == "__main__" :
    batch_size = 8
    n, m = 30, 20

    xyz1 = torch.rand((batch_size, n, 3)).cuda()
    xyz2 = torch.rand((batch_size, m, 3)).cuda()
    #
    # dist1 = torch.zeros(batch_size, n).cuda()
    # dist2 = torch.zeros(batch_size, m).cuda()
    #
    # idx1 = torch.zeros((batch_size, n), dtype=torch.int).cuda()
    # idx2 = torch.zeros((batch_size, m), dtype=torch.int).cuda()
    #
    # chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
    # print(dist1)
    # print(dist2)
    # print(idx1)
    # print(idx2)
    a = ChamferDist()
    print("test")
    print(a(xyz1,xyz2))
