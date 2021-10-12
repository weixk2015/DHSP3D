import os
import time

from data import DataLoader
from models import create_model
from options.train_options import TrainOptions
from util.util import mesh_segmentation, merge_segmentations
from util.util import save_obj, mesh_upsampler
from util.writer import Writer

if __name__ == '__main__':
    opt = TrainOptions().parse()
   # model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.start_epoch,opt.epoch+1):
        opt.cur_epoch = epoch
        obj_path = os.path.join(opt.input_dir, "%s_ep%d.obj" % (opt.obj_filename.split('.obj')[0], opt.cur_epoch))
        gt_path = os.path.join(opt.dataroot, opt.phase, opt.obj_filename)
        opt.batch_size, face_num, dic2olds, overlap, point_num, faces, vs_med, lim , uv ,_= mesh_segmentation(obj_path,laxarr=opt.seg_width)

        if opt.batch_size == -1:
            print("error: too many faces")
            break
        mesh_segmentation(gt_path, parts=opt.batch_size, vs_med=vs_med, lim=lim ,laxnum=opt.seg_width[int(opt.batch_size/4)])

        opt.ninput_edges = int(face_num * 1.55)
        #opt.ninput_edges = int(1.2*edge_counter(faces))
        opt.pool_res = [int(0.9*opt.ninput_edges), int(0.8*opt.ninput_edges)]  #pool_res
        #opt.pool_res = [int(1*opt.ninput_edges), int(1*opt.ninput_edges) ,int(1*opt.ninput_edges)]
        print("#input edges %d,faces %d, pool res " % (opt.ninput_edges,face_num) + str(opt.pool_res))
        model = create_model(opt)  # change create new model

        dataset = DataLoader(opt)  # more than once
        dataset_size = len(dataset)
        print('#training meshes = %d' % dataset_size)
        data_iter = iter(dataset.dataloader)

        dataset.dataset.epoch=epoch
        #data = dataset.dataset.__getitem__(epoch-1)
        data = data_iter.next()
        model.vs_gt=dataset.dataset.vs_gt
        model.vc_gt=dataset.dataset.vc_gt
        model.f_gt=dataset.dataset.f_gt

        epoch_start_time = time.time()
        iter_data_time = time.time()
        model.set_input(data)
        epoch_steps = opt.epoch_steps
        iter_start_time = time.time()
        for step in range(1,epoch_steps+1):
            opt.cur_step = step
            total_steps += 1

            model.optimize_parameters()
            model.update_learning_rate()

            if step % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time)
                iter_start_time = time.time()
                writer.print_current_losses(epoch, step, model.losses, t, time.time() - iter_data_time)
                writer.plot_loss(loss, epoch, step, epoch_steps)

            if step % opt.save_latest_freq == 0:
                print('saving the latest model and .obj (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')
                # for i in range(opt.batch_size):
                #     save_name = "%s_ep%d_part%d_step%d.obj" % (opt.obj_filename.split('.obj')[0],epoch,i,step)
                #     save_obj(model.criterion.pred_coord[i].squeeze(0).detach().cpu().numpy(),model.mesh[i].faces,
                #              opt.result_dir,save_name)
                parts_pts = []
                for i in range(opt.batch_size):
                    parts_pts.append(model.criterion.pred_coord[i].squeeze(0).detach().cpu().numpy())
                out_pts = merge_segmentations(parts_pts, dic2olds, overlap, point_num)
                save_name = "%s_ep%d_step%d.obj" % (opt.obj_filename.split('.obj')[0], epoch,step)
                save_obj(out_pts, faces, opt.result_dir, save_name)
        epoch_steps = int(opt.epoch_steps * 1.2)
        opt.epoch_steps = epoch_steps

        iter_data_time = time.time()
        print('saving the model and .obj at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_network('latest')
        model.save_network(epoch)

        parts_pts = []
        for i in range(opt.batch_size):
            parts_pts.append(model.criterion.pred_coord[i].squeeze(0).detach().cpu().numpy())
        if opt.geo_color:
            parts_vc = []
            for i in range(opt.batch_size):
                parts_vc.append(model.criterion.pred_coord[i].squeeze(0).detach().cpu().numpy())
        else:
            parts_vc = None
        out_pts = merge_segmentations(parts_pts,dic2olds,overlap,point_num,colors=parts_vc)
        save_name = "%s_ep%d_out.obj" % (opt.obj_filename.split('.obj')[0],epoch)
        save_obj(out_pts,faces,opt.output_dir,save_name)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.epoch , time.time() - epoch_start_time))

        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        path = os.path.join("%s_ep%d.obj" % (opt.obj_filename.split('.obj')[0], epoch + 1))

        mesh_upsampler(path, opt.output_dir, opt.input_dir, epoch, nface=len(faces))

    writer.close()

