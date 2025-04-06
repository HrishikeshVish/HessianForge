#!/usr/bin/env python3
import sys
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import trimesh
from sklearn.neighbors import NearestNeighbors

import rclpy
from rclpy.node import Node

##############################################
# Mesh Processing Functions
##############################################

def load_obj(file_path):
    """
    Loads an OBJ file and returns the vertices and faces.
    """
    mesh = trimesh.load_mesh(file_path)
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    return verts, faces

def compute_cotangent_weights(vertices, faces):
    """
    Computes the cotangent weights for the Laplacian matrix.
    
    Args:
        vertices (torch.Tensor): (N, 3) tensor of vertex positions.
        faces (torch.Tensor): (M, 3) tensor of face indices.
    
    Returns:
        L (torch.Tensor): Sparse (N, N) Laplacian matrix with cotangent weights.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

    # Compute edge vectors.
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0 

    # Compute face normals.
    normals = torch.cross(e0, e1, dim=1)

    # Safe cotangent computation.
    def safe_cotangent(a, b):
        cross = torch.cross(a, b, dim=1)
        cross_norm = torch.norm(cross, dim=1)
        dot_product = (a * b).sum(dim=1)
        cot_theta = dot_product / (cross_norm + 1e-8)  # epsilon to avoid division by zero
        return cot_theta

    cot_alpha = safe_cotangent(-e1, e0).squeeze()
    cot_beta  = safe_cotangent(-e2, e1).squeeze()
    cot_gamma = safe_cotangent(-e0, e2).squeeze()
    
    # Build sparse matrix indices and values.
    I = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                   faces[:, 0], faces[:, 1], faces[:, 2]]).cuda()
    J = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                   faces[:, 1], faces[:, 2], faces[:, 0]]).cuda()
    W = 0.5 * torch.cat([cot_alpha, cot_beta, cot_gamma,
                         cot_alpha, cot_beta, cot_gamma]).cuda()

    # Create sparse Laplacian matrix.
    L = torch.sparse_coo_tensor(torch.stack([I, J]).cuda(), W, (vertices.shape[0], vertices.shape[0])).cuda()
    vertices = vertices.cuda()
    W = W.to(vertices.dtype)
    
    # Compute diagonal entries.
    diag_values = torch.zeros(vertices.shape[0], device=vertices.device, dtype=torch.float64).scatter_add_(0, I, W)
    L_diag = torch.sparse_coo_tensor(torch.stack([torch.arange(vertices.shape[0]).cuda(),
                                                   torch.arange(vertices.shape[0]).cuda()]).cuda(),
                                     diag_values.cuda(), (vertices.shape[0], vertices.shape[0]))
    return L - L_diag

def compute_laplacian_magnitude(vertices, faces):
    """
    Computes the Laplacian magnitude for each vertex.
    
    Args:
        vertices (torch.Tensor): (N, 3) tensor of vertex positions.
        faces (torch.Tensor): (M, 3) tensor of face indices.
    
    Returns:
        laplacian_magnitude (torch.Tensor): (N,) tensor of Laplacian magnitudes.
    """
    L = compute_cotangent_weights(vertices, faces)
    delta_v = torch.sparse.mm(L.cuda(), vertices.cuda()).cuda()  # Laplacian displacement
    laplacian_magnitude = torch.norm(delta_v.cuda(), dim=1).cuda()
    return laplacian_magnitude

def expand_neighborhood_and_triangulate(vertices, faces, k=1):
    """
    Expands the neighborhood using KNN and creates new triangulated faces.
    
    Args:
        vertices (torch.Tensor): (N, 3) tensor of vertex positions.
        faces (torch.Tensor): (M, 3) tensor of face indices.
        k (int): Number of new neighbors to consider.
    
    Returns:
        vertices, new_faces (torch.Tensor): The vertices and new triangulated faces.
    """
    print("Expanding Mesh...")
    knn = NearestNeighbors(n_neighbors=2 * k)
    knn.fit(vertices.cpu().numpy())
    distances, indices = knn.kneighbors(vertices.cpu().numpy())
    print("Computed new Neighborhood")
    
    new_faces = set()  # Use a set to avoid duplicate faces.
    current_neighborhood = {i: set() for i in range(vertices.shape[0])}
    for face in faces:
        face_set = tuple(sorted([int(face[0].item()), int(face[1].item()), int(face[2].item())]))
        current_neighborhood[int(face[0].item())].update([int(face[1].item()), int(face[2].item())])
        current_neighborhood[int(face[1].item())].update([int(face[0].item()), int(face[2].item())])
        current_neighborhood[int(face[2].item())].update([int(face[0].item()), int(face[1].item())])
        new_faces.add(face_set)
    
    from itertools import combinations
    print("...Triangulating and generating new Faces...")
    for i in tqdm(range(vertices.shape[0])):
        neighbors = indices[i]
        # Filter out neighbors already connected.
        filtered_neighbors = [v for v in neighbors if v not in current_neighborhood[i]]
        if len(filtered_neighbors) > k:
            filtered_neighbors = filtered_neighbors[:k]
        assert len(filtered_neighbors) > 0, f"Vertex {i} has no valid neighbors after filtering."
        for v1, v2 in combinations(filtered_neighbors, 2):
            face_tuple = tuple(sorted([i, v1, v2]))
            new_faces.add(face_tuple)
    print("Triangulated the mesh")
    new_faces = torch.tensor([list(face) for face in new_faces], dtype=torch.int64)
    print("Number of Faces in the Triangulated Mesh:", new_faces.shape)
    print("Number of Faces in Original Mesh:", faces.shape)
    return vertices, new_faces

def minimize_laplacian_energy(vertices, faces, num_iters=1500, eta=0.001, expand_neighborhood=False, is_hessian_energy=False):
    """
    Minimize Laplacian energy to smooth the mesh.
    
    Args:
        vertices (torch.Tensor): (N, 3) Vertex positions.
        faces (torch.Tensor): (M, 3) Face indices.
        num_iters (int): Number of iterations.
        eta (float): Learning rate.
        expand_neighborhood (bool): Whether to expand the neighborhood.
        is_hessian_energy (bool): Whether to use second-order (Hessian) energy.
    
    Returns:
        torch.Tensor: Optimized vertex positions.
    """
    if expand_neighborhood:
        _, cells = expand_neighborhood_and_triangulate(vertices, faces)
        cells = cells.cuda()
    else:
        cells = torch.from_numpy(faces).long().cuda()
    V = vertices.clone()
    print("Optimizing Mesh")
    progress_bar = tqdm(range(num_iters))
    for _ in progress_bar:
        L = compute_cotangent_weights(V, cells).cuda()
        V = V.cuda()
        delta_V = torch.sparse.mm(L, V)
        if is_hessian_energy:
            bi_laplacian_V = torch.sparse.mm(L, delta_V)
            hessian_energy = torch.sum(bi_laplacian_V**2, dim=1)
            laplacian_magnitude = hessian_energy
        else:
            laplacian_magnitude = torch.norm(delta_V, dim=1)
        
        # Compute per-vertex neighborhood average.
        coalesced_L = L.coalesce()
        I, J = coalesced_L.indices()
        neighbor_sum = torch.zeros_like(V).scatter_add_(0, I[:, None].expand(-1, 3), V[J])
        neighbor_count = torch.zeros(V.shape[0], device=V.device, dtype=V.dtype).scatter_add_(0, I, torch.ones_like(I, dtype=V.dtype))
        valid_mask = neighbor_count > 0
        neighbor_avg = torch.zeros_like(V)
        neighbor_avg[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask][:, None]

        # Set weights based on local curvature.
        laplacian_magnitude[laplacian_magnitude > 1] = 0
        weight = torch.exp(-laplacian_magnitude)
        V = V + eta * weight[:, None] * (neighbor_avg - V)
        progress_bar.set_postfix({
            'Mean Laplacian': laplacian_magnitude.mean().item(),
            'Max Laplacian': laplacian_magnitude.max().item()
        })

    return V

##############################################
# ROS2 Node for Mesh Refinement
##############################################

class MeshRefinementNode(Node):
    def __init__(self):
        super().__init__('mesh_refinement_node')
        # Declare parameters to configure the process.
        self.declare_parameter('mesh_name', 'tower')
        self.declare_parameter('obj_folder', 'obj_meshes')
        self.declare_parameter('num_iters', 1500)
        self.declare_parameter('eta', 0.001)
        self.declare_parameter('is_hessian_energy', False)
        self.declare_parameter('expand_neighborhood', False)

        self.mesh_name = self.get_parameter('mesh_name').value
        self.obj_folder = self.get_parameter('obj_folder').value
        self.num_iters = self.get_parameter('num_iters').value
        self.eta = self.get_parameter('eta').value
        self.is_hessian_energy = self.get_parameter('is_hessian_energy').value
        self.expand_neighborhood = self.get_parameter('expand_neighborhood').value

        self.get_logger().info("Starting mesh refinement for mesh: " + self.mesh_name)
        self.run_refinement()

    def run_refinement(self):
        # Load mesh from OBJ.
        obj_path = f"{self.obj_folder}/{self.mesh_name}.obj"
        verts_np, faces_np = load_obj(obj_path)
        self.get_logger().info(f"Loaded mesh with {verts_np.shape[0]} vertices and {faces_np.shape[0]} faces.")

        # Compute initial Laplacian magnitude.
        verts_tensor = torch.from_numpy(verts_np)
        faces_tensor = torch.from_numpy(faces_np).long()
        laplacian = compute_laplacian_magnitude(verts_tensor, faces_tensor)
        self.get_logger().info(f"Initial Laplacian: max {laplacian.max().item()}, min {laplacian.min().item()}, mean {laplacian.mean().item()}")

        # Clamp high values for stability.
        laplacian[laplacian > 1] = 0
        self.get_logger().info("Clamped high Laplacian values.")

        # Save initial mesh info.
        torch.save({"v": verts_tensor, "f": faces_tensor, "L": laplacian}, f"{self.mesh_name}.pt")
        
        # Run minimization to refine the mesh.
        V_optimized = minimize_laplacian_energy(verts_tensor, faces_np, num_iters=self.num_iters,
                                                eta=self.eta, expand_neighborhood=self.expand_neighborhood,
                                                is_hessian_energy=self.is_hessian_energy)
        new_laplacian = compute_laplacian_magnitude(V_optimized, faces_tensor)
        self.get_logger().info(f"Refined Laplacian: max {new_laplacian.max().item()}, min {new_laplacian.min().item()}, mean {new_laplacian.mean().item()}")
        new_laplacian[new_laplacian > 1] = 0
        torch.save({"v": V_optimized, "f": faces_tensor, "L": new_laplacian}, f"{self.mesh_name}_refined_{self.num_iters}.pt")
        
        # Convert optimized vertices to numpy for visualization.
        verts_refined = V_optimized.detach().cpu().numpy()
        refined_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts_refined),
            o3d.utility.Vector3iVector(faces_np)
        )
        self.get_logger().info("Mesh refinement complete. Refined mesh saved as a PyTorch checkpoint.")
        # Optionally, you could publish or visualize the mesh using Open3D here.
        # For example:
        # o3d.visualization.draw_geometries([refined_mesh])
        # Since this is a batch process, we shut down once complete.
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MeshRefinementNode()
    # For this batch process, we don't need to spin.
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
