# Deep Hybrid self-prior

This is a PyTorch implementation of the ICCV2021 paper
"Deep Hybrid Self-Prior for Full 3D Mesh Generation".

In this work, We present a deep learning pipeline that leverages 
network self-prior to recover a full 3D model consisting of both 
a triangular mesh and a texture map from the colored 3D point cloud. 
Different from previous methods either exploiting 2D self-prior for 
image editing or 3D self-prior for pure surface reconstruction, we 
propose to exploit a novel hybrid 2D-3D self-prior in deep neural 
networks to significantly improve the geometry quality and produce 
a high-resolution texture map, which is typically missing from the 
output of commodity-level 3D scanners. Experiments show that, 
without the need of any additional training data, 
our method recovers the 3D textured mesh model of high quality 
from sparse input, and outperforms the state-of-the-art methods 
in terms of both the geometry and texture quality.

Please check the [paper](https://arxiv.org/abs/2108.08017)
and the [project webpage](https://yqdch.github.io/DHSP3D/) for more details.

If you have any question, please contact Xingkui Wei <xkwei19@fudan.edu.cn>.


#### Citation

If you use this code for any purpose, please consider citing:
```
@inproceedings{wei2021deep,
title={Deep Hybrid Self-Prior for Full 3D Mesh Generation},
author={Wei, Xingkui and Chen, Zhengqing and Fu, Yanwei and Cui, Zhaopeng and Zhang, Yinda},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={5805--5814},
year={2021}
}
```


## Requirements

Building and using requires the following libraries and programs
- python=3.6.8
- cython=0.27.3
- pytorch=1.2.0
- numpy=1.15.0
- matplotlib=3.0.3
- pip
- tensorboardX
- pytest==5.1.1

The versions match the configuration we have tested on an ubuntu 16.04 system.

## Running

**under constructing. We show a example for generate textured mesh from point cloud(dog.obj) below.**

cp dog.obj datasets/train/

1. generate convex hull via meshlab, install manifold package

dog.obj -> dog_convex.obj
install Manifold: https://github.com/hjwdzh/Manifold.git
./manifold dog_convex.obj output.obj
./simplify -i output.obj -o dog_ep1.obj -f 2000 
mkdir datasets/result/dog/input
cp dog_ep1.obj datasets/result/dog/input/

2. run Point2Mesh (3D prior)

python train.py --dataroot datasets --name dog 
--arch meshunet --obj_filename dog.obj 
--resblocks 2 --lr 0.001 --weight_edge_loss 
0.2 --epoch_steps 2000 --epoch 4

cp datasets/result/dog/dog_ep4_step2000.obj dog_p.obj

3. run UVflatten algorithm OptCuts

install OptCuts: https://github.com/liminchen/OptCuts 

OptCuts/build/OptCuts_bin 10 dog_p.obj  0.999 1 0 4.15 1 0

cp OptCuts/output/dog_p_Tutte_0.999_1_OptCuts/finalResult_mesh_normalizedUV.obj dog_p_uv.obj

4. Geometric Projection and Completion(2D prior)

python project_geo.py dog

python im_complete.py dog

python back_proj.py dog

5. run Point2Mesh  (3D prior)
python train.py --dataroot datasets --name dogn 
--arch meshunet --obj_filename dog.obj 
--resblocks 2 --lr 0.001 --weight_edge_loss 
0.2 --epoch_steps 1200 --epoch 1

cp datasets/result/dogn/dog_ep1_step1200.obj dog_f.obj

Step (3-5) could be repeated as iterations
 
6. run UVflatten algorithm OptCuts

OptCuts/build/OptCuts_bin 10 dog_f.obj  0.999 1 0 4.15 1 0

cp OptCuts/output/dog_f_Tutte_0.999_1_OptCuts/finalResult_mesh_normalizedUV.obj dog_f_uv.obj

7. Texture Projection and Completion (2D prior)

python project_texture.py dog

python im_complete_text.py dog

cp output/doghr.jpg dog.jpg

8. Full Mesh Generation

modify dog_f_uv.obj: add mtllib dog.mtl

create dog.mtl and write:
newmtl material_0
Ka 0.200000 0.200000 0.200000
Kd 0.749020 0.749020 0.749020
Ks 1.000000 1.000000 1.000000
Tr 1.000000
illum 2
Ns 0.000000
map_Kd dog.jpg

dog_f_uv.obj is the final textured mesh

## Acknowledgments

The implementation codes borrows heavily from [MeshCNN](https://github.com/ranahanocka/MeshCNN). Thanks for the sharing.
