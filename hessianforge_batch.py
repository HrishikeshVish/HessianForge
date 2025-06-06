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
import wandb
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import  VCycleAggregator
from dataset.lidar_dataset import LiDARDataset


def run_mapping_batch():
    """
    Runs the mapping process over a batch of LiDAR frames.
    
    This function sets up the experiment, loads data and models, and performs
    training over several iterations to reconstruct a scene from LiDAR point clouds.
    """
    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("Please provide the path to the config file.\nTry: python shine_batch.py xxx/xxx_config.yaml")
    
    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path)

    dev = config.device

    # Initialize feature octree and physics engine
    octree = FeatureOctree(config)
    phys_eng = VCycleAggregator(L=4).cuda()

    # Load pretrained model if specified
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        phys_eng.load_state_dict(loaded_model["phys_eng"])
        print("Pretrained decoder loaded")
        if 'feature_octree' in loaded_model:
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    # Load dataset and initialize mesher
    dataset = LiDARDataset(config, octree)
    mesher = Mesher(config, octree, phys_eng=phys_eng)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    if config.o3d_vis_on:
        vis = MapVisualizer()

    print("Load, preprocess and sample data")
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0):
            continue
        t0 = get_time()
        dataset.process_frame(frame_id)
        t1 = get_time()

    # Prepare parameters for optimization
    octree_feat = list(octree.parameters())
    decoder_param = list(phys_eng.parameters())
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev) * 1.0)
    sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale

    pc_map_path = run_path + '/map/pc_map_down.ply'
    dataset.write_merged_pc(pc_map_path)

    opt = setup_optimizer(config, octree_feat, decoder_param, sigma_size)
    octree.print_detail()

    require_gradient = config.normal_loss_on or config.ekional_loss_on or \
                       config.proj_correction_on or config.consistency_loss_on

    print("Begin mapping")
    cur_base_lr = config.lr
    progress_bar = tqdm(range(config.iters))

    for iter in progress_bar:
        T0 = get_time()
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)

        # Get training batch
        if config.ray_loss:
            coord, sample_depth, ray_depth, normal_label, sem_label, weight = dataset.get_batch()
        else:
            coord, sdf_label, origin, ts, normal_label, sem_label, weight = dataset.get_batch()

        if require_gradient:
            coord.requires_grad_(True)

        T1 = get_time()
        feature = octree.query_feature(coord, return_cat=True)
        T2 = get_time()
        pred = phys_eng(feature)
        surface_mask = weight > 0
        T3 = get_time()

        if require_gradient:
            g = get_gradient(coord, pred) * sigma_sigmoid
            g_hess = g / sigma_sigmoid

            if config.hessian_loss_on:
                hessian = torch.zeros(pred.shape[0], 3, 3, device=coord.device)
                for i in range(3):
                    grad_fi = g_hess[:, i]
                    hess_i = torch.autograd.grad(
                        grad_fi, coord,
                        grad_outputs=torch.ones_like(grad_fi),
                        create_graph=True
                    )[0]
                    hessian[:, i, :] = hess_i

        if config.proj_correction_on:
            cos = torch.abs(F.cosine_similarity(g, coord - origin))
            cos[~surface_mask] = 1.0
            sdf_label = sdf_label * cos

        if config.consistency_loss_on:
            near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count, coord.shape[0]),), device=dev)
            shift_scale = config.consistency_range * config.scale
            random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            coord_near = coord + random_shift
            coord_near = coord_near[near_index, :]
            coord_near.requires_grad_(True)
            feature_near = octree.query_feature(coord_near)
            pred_near = phys_eng(feature_near)
            g_near = get_gradient(coord_near, pred_near) * sigma_sigmoid

        cur_loss = 0.0

        # Compute main loss
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

        # Optional losses
        eikonal_loss = 0.
        hessian_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean()
            cur_loss += config.weight_e * eikonal_loss

        if config.hessian_loss_on:
            hessian_loss = sdf_hessian_loss(pred, coord, hessian, surface_mask,
                                            hessian_norm_scale=config.hessian_norm_scale,
                                            hessian_loss_scale=config.hessian_loss_scale)
            cur_loss += config.weight_h * hessian_loss

        consistency_loss = 0.
        if config.consistency_loss_on:
            consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
            cur_loss += config.weight_c * consistency_loss

        normal_loss = 0.
        if config.normal_loss_on:
            g_direction = g / g.norm(2, dim=-1)
            normal_diff = g_direction - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
            cur_loss += config.weight_n * normal_loss

        T4 = get_time()
        opt.zero_grad(set_to_none=True)
        cur_loss.backward()
        opt.step()
        T5 = get_time()

        progress_bar.set_postfix({
            "iter": iter,
            "loss": cur_loss.item(),
            "eikonal": eikonal_loss.item(),
            "hessian": hessian_loss.item()
        })

        # Logging with wandb
        if config.wandb_vis_on:
            wandb_log_content = {
                'iter': iter,
                'loss/total_loss': cur_loss,
                'loss/eikonal_loss': eikonal_loss,
                'loss/normal_loss': normal_loss,
                'timing(s)/load': T1 - T0,
                'timing(s)/get_indices': T2 - T1,
                'timing(s)/inference': T3 - T2,
                'timing(s)/cal_loss': T4 - T3,
                'timing(s)/back_prop': T5 - T4,
                'timing(s)/total': T5 - T0,
                'para/sigma': sigma_size,
            }

            if config.ray_loss:
                wandb_log_content['loss/render_loss'] = dr_loss
            else:
                wandb_log_content['loss/sdf_loss'] = sdf_loss
                wandb_log_content['loss/consistency_loss'] = consistency_loss
                # sem_loss might be defined in an extended context
                if 'sem_loss' in locals():
                    wandb_log_content['loss/sem_loss'] = sem_loss

            wandb.log(wandb_log_content)

        # Checkpoint saving logic is assumed to continue after this


if __name__ == "__main__":
    run_mapping_batch()