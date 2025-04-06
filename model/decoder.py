import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import SHINEConfig


class VCycleAggregator(nn.Module):
    def __init__(self, L, feature_dim=32):
        super().__init__()
        self.L = L
        self.feature_dim = feature_dim

        # Define MLPs for fine-to-coarse aggregation and coarse-to-fine interpolation
        self.fine_to_coarse_aggregator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 8)
            ) 
            for _ in range(L - 1)])
        self.coarse_to_fine_interpolator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 8)
            ) 
            for _ in range(L - 1)])

        self.layer_downsample = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(8, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
        
    def forward(self, octree_feats):
        """
        V-Cycle aggregation with Bézier kernel integration.
        Args:
            octree_feats (torch.Tensor): Input hierarchical features of shape [N, 8, L].
        Returns:
            torch.Tensor: Refined hierarchical features of shape [N, 8, L].
        """


        N, features, corners, L = octree_feats.shape
        assert corners == 8, "Input octree_feats must have 8 corner features per voxel."

        
        # Initialize hierarchical features
        hierarchical_features = [octree_feats[:, :, :, i] for i in range(L)]  # Each: [N, 8, 8]

        # Fine-to-coarse refinement
        for i in range(L - 1, 0, -1):  
            coarse_features = hierarchical_features[i - 1]  # Shape: [N, 8, 8]
            fine_features = hierarchical_features[i]  # Shape: [N, 8, 8]

            aggregated_features = torch.mean(fine_features, dim=2, keepdim=True)  # Shape: [N,8, 1]
            refined_coarse = self.fine_to_coarse_aggregator[i - 1](aggregated_features)  # Shape: [N, 8, 8]


            hierarchical_features[i - 1] += refined_coarse  # Shape: [N, 8, 8]

        # Coarse-to-fine refinement
        for i in range(L - 1):
            coarse_features = hierarchical_features[i]  # Shape: [N, 8]
            fine_features = hierarchical_features[i + 1]  # Shape: [N, 8]
            

            aggregated_features = torch.mean(coarse_features, dim=2, keepdim=True)  # Shape: [N, 8, 1]

            interpolated_features = self.coarse_to_fine_interpolator[i](aggregated_features)  # Shape: [N,8, 8]

            hierarchical_features[i + 1] += interpolated_features  # Shape: [N, 8, 8]

        global_features = torch.zeros_like(hierarchical_features[0])
        for i in range(L):
            global_features += hierarchical_features[i] 
        
        global_features = torch.sum(global_features, dim=1, keepdim=False)
        sdf = self.final_mlp(global_features)
        # Stack results back to original shape [N, 8, L]
        return sdf.squeeze(-1)