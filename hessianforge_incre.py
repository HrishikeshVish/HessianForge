"""
This codebase references several components from the SHINE-Mapping project:
https://github.com/PRBonn/SHINE_mapping

SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations
by Xingguang Zhong, Yue Pan, Jens Behley, and Cyrill Stachniss.
Presented at the IEEE International Conference on Robotics and Automation (ICRA), 2023.

Citation:
@inproceedings{zhong2023icra,
  title={SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations},
  author={Zhong, Xingguang and Pan, Yue and Behley, Jens and Stachniss, Cyrill},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}

We thank the authors for making their work open source.
"""

import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import shutil

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.incre_learning import cal_feature_importance
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import VCycleAggregator
from dataset.lidar_dataset import LiDARDataset

def run_shine_mapping_incremental():

    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
        )

    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path)  # copy the config file to the result folder

    dev = config.device

    # initialize the feature octree
    octree = FeatureOctree(config)

    # add the physical engine module (VCycleAggregator)
    phys_eng = VCycleAggregator(L=4).to(dev)

    # Load the decoder model (and physical engine) if provided
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        
        phys_eng.load_state_dict(loaded_model["phys_eng"])
        print("Pretrained decoder loaded")
        if config.semantic_on:
            raise ValueError("Semantic decoder is not supported in this version.")
        if 'feature_octree' in loaded_model.keys():
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    # dataset
    dataset = LiDARDataset(config, octree)

    # mesh reconstructor (now passing phys_eng)
    mesher = Mesher(config, octree, phys_eng=phys_eng)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Non-blocking visualizer
    if config.o3d_vis_on:
        vis = MapVisualizer()

    # collect learnable parameters
    phys_eng_param = list(phys_eng.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev) * 1.0)
    # fixed sigma for SDF prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale

    processed_frame = 0
    total_iter = 0

    if config.continual_learning_reg:
        config.loss_reduction = "sum"

    # if any of these losses are on, we require gradients for the input coordinates
    if config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on:
        require_gradient = True
    else:
        require_gradient = False

    # for each frame (incremental update)
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or
            frame_id % config.every_frame != 0):
            continue

        vis_mesh = False

        if processed_frame == config.freeze_after_frame:
            print("Freeze the decoder")
            freeze_model(phys_eng)
            if config.semantic_on:
                raise ValueError("Semantic decoder is not supported in this version.")

        T0 = get_time()
        # process frame and update the octree (data pool update depending on continual learning)
        local_data_only = False
        dataset.process_frame(frame_id, incremental_on=config.continual_learning_reg or local_data_only)

        octree_feat = list(octree.parameters())
        opt = setup_optimizer(config, octree_feat, phys_eng_param, sigma_size)
        octree.print_detail()

        T1 = get_time()

        # Here we run training iterations for the current frame
        for iter in tqdm(range(config.iters)):
            # -- Data loading --
            # Branch based on ray loss setting; note that when not using ray loss, we expect additional outputs.
            if config.ray_loss:
                # get_batch() should return: coord, sample_depth, ray_depth, normal_label, sem_label, weight
                coord, sample_depth, ray_depth, normal_label, sem_label, weight = dataset.get_batch()
            else:
                # get_batch() returns: coord, sdf_label, origin, ts, normal_label, sem_label, weight
                coord, sdf_label, origin, ts, normal_label, sem_label, weight = dataset.get_batch()

            if require_gradient:
                coord.requires_grad_(True)

            # -- Feature extraction --
            # now also concatenate hierarchical features (as in shine_batch.py)
            feature = octree.query_feature(coord, return_cat=True)

            # -- Prediction --
            # use phys_eng when not time-conditioned
            if not config.time_conditioned:
                pred = phys_eng(feature)
            else:
                raise ValueError("Time-conditioned prediction is not supported in this version.")

            if config.semantic_on:
                raise ValueError("Semantic decoder is not supported in this version.")

            # -- Gradient (and Hessian) computation --
            surface_mask = weight > 0
            if require_gradient:
                g = get_gradient(coord, pred) * sigma_sigmoid
                # Compute Hessian
                g_hess = g / sigma_sigmoid
                if(config.hessian_loss_on):
                    hessian = torch.zeros(pred.shape[0], 3, 3, device=coord.device)
                    grad_outputs = torch.ones_like(pred)
                    for i in range(3):
                        grad_fi = g_hess[:, i]
                        hess_i = torch.autograd.grad(grad_fi, coord, grad_outputs=torch.ones_like(grad_fi), create_graph=True)[0]
                        hessian[:, i, :] = hess_i

            # -- Optional projection correction --
            if config.proj_correction_on and (not config.ray_loss):
                cos = torch.abs(F.cosine_similarity(g, coord - origin))
                cos[~surface_mask] = 1.0
                sdf_label = sdf_label * cos

            # -- Consistency loss (if enabled) --
            if config.consistency_loss_on:
                near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count, coord.shape[0]),), device=dev)
                shift_scale = config.consistency_range * config.scale
                random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
                coord_near = coord + random_shift
                coord_near = coord_near[near_index, :]
                coord_near.requires_grad_(True)
                feature_near = octree.query_feature(coord_near, return_cat=True)
                # For consistency, use the same geo decoder branch (note: this branch is not time-conditioned)
                if not config.time_conditioned:
                    pred_near = phys_eng(feature_near)
                else:
                    raise ValueError("Time-conditioned prediction is not supported in this version.")
                g_near = get_gradient(coord_near, pred_near) * sigma_sigmoid

            # -- Loss computation --
            cur_loss = 0.
            if config.ray_loss:
                pred_occ = torch.sigmoid(pred / sigma_size)
                pred_ray = pred_occ.reshape(config.bs, -1)
                sample_depth = sample_depth.reshape(config.bs, -1)
                if config.main_loss_type == "dr":
                    dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=False)
                elif config.main_loss_type == "dr_neus":
                    dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=True)
                cur_loss += dr_loss
            else:
                weight = torch.abs(weight)
                if config.main_loss_type == "sdf_bce":
                    sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction)
                elif config.main_loss_type == "sdf_l1":
                    sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=False)
                elif config.main_loss_type == "sdf_l2":
                    sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=True)
                cur_loss += sdf_loss

            # Optional eikonal and Hessian regularization loss
            eikonal_loss = 0.
            hessian_loss = 0.
            if config.ekional_loss_on and require_gradient:
                eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean()
                cur_loss += config.weight_e * eikonal_loss
            if config.hessian_loss_on:
                hessian_loss = sdf_hessian_loss(pred, coord, hessian, surface_mask, hessian_norm_scale=config.hessian_norm_scale, hessian_loss_scale=config.hessian_loss_scale)
                cur_loss += config.weight_h * hessian_loss

            # Consistency loss
            if config.consistency_loss_on:
                consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
                cur_loss += config.weight_c * consistency_loss

            # Normal loss
            normal_loss = 0.
            if config.normal_loss_on and require_gradient:
                g_direction = g / g.norm(2, dim=-1)
                normal_diff = g_direction - normal_label
                normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
                cur_loss += config.weight_n * normal_loss

            # Semantic classification loss
            sem_loss = 0.
            if config.semantic_on:
                raise ValueError("Semantic decoder is not supported in this version.")

            opt.zero_grad(set_to_none=True)
            cur_loss.backward()
            opt.step()

            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {
                    'iter': total_iter,
                    'loss/total_loss': cur_loss,
                    'loss/sdf_loss' if not config.ray_loss else 'loss/render_loss': sdf_loss if not config.ray_loss else dr_loss,
                    'loss/eikonal_loss': eikonal_loss,
                    'loss/hessian_loss': hessian_loss,
                    'loss/consistency_loss': consistency_loss if config.consistency_loss_on else 0,
                    'loss/normal_loss': normal_loss,
                    'loss/sem_loss': sem_loss
                }
                wandb.log(wandb_log_content)

        # After frame training, if using continual learning regularization, update importance
        if config.continual_learning_reg:
            opt.zero_grad(set_to_none=True)
            cal_feature_importance(dataset, octree, phys_eng, sigma_sigmoid, config.bs,
                                   config.cal_importance_weight_down_rate, config.loss_reduction)

        T2 = get_time()

        # Mesh reconstruction (optional) using the current map
        if processed_frame == 0 or (processed_frame + 1) % config.mesh_freq_frame == 0:
            print("Begin mesh reconstruction from the implicit map")
            vis_mesh = True
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id + 1) + ".ply"
            map_path = run_path + '/map/sdf_map_frame_' + str(frame_id + 1) + ".ply"
            if config.mc_with_octree:
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
            else:
                if config.mc_local:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.cur_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        T3 = get_time()

        if config.o3d_vis_on:
            if vis_mesh:
                cur_mesh.transform(dataset.begin_pose_inv)
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref, cur_mesh)
            else:
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref)

        if config.wandb_vis_on:
            wandb_log_content = {
                'frame': processed_frame,
                'timing(s)/preprocess': T1 - T0,
                'timing(s)/mapping': T2 - T1,
                'timing(s)/reconstruct': T3 - T2
            }
            wandb.log(wandb_log_content)

        processed_frame += 1

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_shine_mapping_incremental()
