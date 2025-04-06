import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import trimesh

def load_obj(file_path):
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

    # Compute edge vectors
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0 
    

    # Compute face normals (cross product of two edge vectors)
    normals = torch.cross(e0, e1, dim=1)

    # Compute cotangent weights safely
    def safe_cotangent(a, b):
        cross = torch.cross(a, b, dim=1)
        cross_norm = torch.norm(cross, dim=1)
        dot_product = (a * b).sum(dim=1)
        cot_theta = dot_product / (cross_norm + 1e-8)  # Add small epsilon to prevent division by zero
        return cot_theta

    cot_alpha = safe_cotangent(-e1, e0).squeeze()
    cot_beta  = safe_cotangent(-e2, e1).squeeze()
    cot_gamma = safe_cotangent(-e0, e2).squeeze()
    

    # Construct sparse matrix indices and values efficiently
    I = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],faces[:, 0], faces[:, 1], faces[:, 2]]).cuda()
    J = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],faces[:, 1], faces[:, 2], faces[:, 0]]).cuda()
    W = 0.5 * torch.cat([cot_alpha, cot_beta, cot_gamma,cot_alpha, cot_beta, cot_gamma]).cuda()


    # Create sparse Laplacian matrix
    L = torch.sparse_coo_tensor(torch.stack([I, J]).cuda(), W, (vertices.shape[0], vertices.shape[0])).cuda()
    vertices = vertices.cuda()
    W = W.to(vertices.dtype)

    
    
    # Compute diagonal entries in a fully vectorized way
    diag_values = torch.zeros(vertices.shape[0], device=vertices.device, dtype=torch.float64).scatter_add_(0, I, W)
        
    L_diag = torch.sparse_coo_tensor(torch.stack([torch.arange(vertices.shape[0]).cuda(), torch.arange(vertices.shape[0]).cuda()]).cuda(), diag_values.cuda(), (vertices.shape[0], vertices.shape[0]))

    return L - L_diag

def compute_laplacian_magnitude(vertices, faces):
    """
    Computes the Laplacian magnitude for each vertex to detect bumps.
    
    Args:
        vertices (torch.Tensor): (N, 3) tensor of vertex positions.
        faces (torch.Tensor): (M, 3) tensor of face indices.

    Returns:
        laplacian_magnitude (torch.Tensor): (N,) tensor of Laplacian magnitudes.
    """
    L = compute_cotangent_weights(vertices, faces)
    
    # Apply Laplacian operator
    delta_v = torch.sparse.mm(L.cuda(), vertices.cuda()).cuda()  # Laplacian displacement

    # Compute magnitude of Laplacian
    laplacian_magnitude = torch.norm(delta_v.cuda(), dim=1).cuda()  # (N,) shape

    return laplacian_magnitude

# Function to expand the neighborhood and triangulate
def expand_neighborhood_and_triangulate(vertices, faces, k=1):
    # Step 1: Compute KNN graph (2*K neighbors)
    print("Expanding Mesh...")
    
    knn = NearestNeighbors(n_neighbors=2 * k)
    knn.fit(vertices.cpu().numpy())
    distances, indices = knn.kneighbors(vertices.cpu().numpy())
    print("Computed new Neighborhood")

    # Step 2: Generate expanded set of vertices (original + neighbors)
    new_faces = set()  # Use a set to automatically handle duplicate faces
    vertex_neighbors = {i: [] for i in range(vertices.shape[0])}  # Create a map to store neighbors for each vertex
    current_neighborhood = {i:set() for i in range(vertices.shape[0])}
    # Step 3: Add the original faces to the new_faces set
    for face in faces:
        face_set = tuple(sorted([face[0].item(), face[1].item(), face[2].item()]))
        current_neighborhood[face[0].item()].add(face[1].item())
        current_neighborhood[face[0].item()].add(face[2].item())
        current_neighborhood[face[1].item()].add(face[2].item())
        current_neighborhood[face[1].item()].add(face[0].item())
        current_neighborhood[face[2].item()].add(face[1].item())
        current_neighborhood[face[2].item()].add(face[0].item())
        assert len(face_set) == 3, "Faces should have 3 vertices"
        new_faces.add(face_set)
    from itertools import combinations
    # Step 4: Add new faces and efficiently connect vertices
    print("...Triangulating and generating new Faces...")
    for i in tqdm(range(vertices.shape[0])):
        neighbors = indices[i]
        
        # Step 4a: Filter out neighbors that are already connected via an edge
        filtered_neighbors = [v for v in neighbors if v not in current_neighborhood[i]]
        
        # Step 4b: If filtered neighbors are greater than k, select top k
        if len(filtered_neighbors) > k:
            filtered_neighbors = filtered_neighbors[:k]
        # Assertion: Ensure filtered_neighbors contains at least one valid neighbor
        assert len(filtered_neighbors) > 0, f"Vertex {i} has no valid neighbors after filtering."
        # Step 4d: Store neighbors in the hashmap
        #vertex_neighbors[i] = filtered_neighbors
        faces_set = set()
        for v1, v2 in combinations(filtered_neighbors, 2):
            tup = tuple(sorted([i, v1, v2]))
            assert len(tup) == 3, "Faces should have 3 vertices"
            faces_set.add(tup)
        new_faces.update(faces_set)
        #new_faces.update({frozenset([i, v1, v2]) for v1, v2 in combinations(filtered_neighbors, 2) if v1!=i and v2!=i and v1!=v2 })
    print("Triangulated the mesh")
    # Convert set of faces to tensor format for consistency
    
    new_faces = torch.tensor([list(face) for face in new_faces], dtype=torch.int64)
    print("Number of Faces in the Triangulated Mesh ", new_faces.shape)
    print("Number of Faces in Original Mesh: ", faces.shape)

    return vertices, new_faces

# # Example Usage
# N = 1000
# points = torch.randn(N, 3)  # Random 3D points
# sdf = torch.randn(N)        # Random SDF values

# bump_points, bump_indices = detect_bumps(sdf, points, threshold_factor=1.5, k=6)
# visualize_bumps(points, bump_points)

def minimize_laplacian_energy(vertices, faces, L, num_iters=1500, eta=0.001, expand_neighborhood=False, is_hessian_energy=False):
    """
    Minimize Laplacian energy to smooth the mesh.

    Args:
        vertices (torch.Tensor): (N, 3) Vertex positions.
        faces (torch.Tensor): (M, 3) Face indices.
        num_iters (int): Number of iterations.
        eta (float): Learning rate (step size).

    Returns:
        torch.Tensor: Optimized vertex positions.
    """
    # Compute initial Laplacian (assumes fixed topology)
    # Clone vertices to avoid modifying input
    if(expand_neighborhood == True):
        _, cells = expand_neighborhood_and_triangulate(vertices, faces)
        cells = cells.cuda()
    else:
        cells = torch.from_numpy(faces).long().cuda()
    V = vertices.clone()
    print("Optimizing Mesh")
    progress_bar = tqdm(range(num_iters))
    for _ in progress_bar:
        L = compute_cotangent_weights(V, cells)
        L = L.cuda()
        V = V.cuda()
        delta_V = torch.sparse.mm(L, V)  # Compute Laplacian displacement
        if(is_hessian_energy == True):
            bi_laplacian_V = torch.sparse.mm(L, delta_V)  # Second Laplacian (L² V)
            # Compute L²-Hessian energy as sum of squared norms
            hessian_energy = torch.sum(bi_laplacian_V**2, dim=1)
            laplacian_magnitude = hessian_energy
        else:
            laplacian_magnitude = torch.norm(delta_V, dim=1)  # Curvature estimation
        
        # Compute per-vertex neighborhood average position
        coalsed_L = L.coalesce() # Get edges from sparse matrix
        I, J = coalsed_L.indices()
        neighbor_sum = torch.zeros_like(V).scatter_add_(0, I[:, None].expand(-1, 3), V[J])
        neighbor_count = torch.zeros(V.shape[0], device=V.device, dtype=V.dtype).scatter_add_(0, I, torch.ones_like(I, dtype=V.dtype))
        
        # Avoid division by zero for isolated vertices
        valid_mask = neighbor_count > 0
        neighbor_avg = torch.zeros_like(V)
        neighbor_avg[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask][:, None]

        # Compute adaptive step size: larger for high-curvature areas
        #weight = torch.exp(-laplacian_magnitude)  # Weight directly proportional to curvature
        laplacian_magnitude[laplacian_magnitude>1] = 0
        #print(laplacian_magnitude.max(), laplacian_magnitude.min(), laplacian_magnitude.mean())
        weight = torch.exp(-laplacian_magnitude)
        V = V + eta * weight[:, None] * (neighbor_avg - V)
        progress_bar.set_postfix({'Mean Laplacian':laplacian_magnitude.mean().item(), 'Max Laplacian':laplacian_magnitude.max().item()})

    return V

mesh_name = 'tower'
num_iters = 1500
verts, faces = load_obj(f'obj_meshes/{mesh_name}.obj')
laplacian = compute_laplacian_magnitude(torch.from_numpy(verts), torch.from_numpy(faces).long())
print(verts.shape)
print(laplacian.shape)
print(laplacian.max(), laplacian.min(), laplacian.mean())
laplacian[laplacian>1] = 0
print(laplacian.max(), laplacian.min(), laplacian.mean())
verts = torch.from_numpy(verts)
torch.save({"v":verts, "f":torch.from_numpy(faces), "L":laplacian}, f"{mesh_name}.pt")
V = minimize_laplacian_energy(verts, faces, laplacian, num_iters=num_iters, eta=0.001, is_hessian_energy=False)
new_laplacian = compute_laplacian_magnitude(V, torch.from_numpy(faces).long())
print(new_laplacian.max(), new_laplacian.min(), new_laplacian.mean())
new_laplacian[new_laplacian>1] = 0
print(new_laplacian.max(), new_laplacian.min(), new_laplacian.mean())
torch.save({"v":V, "f":torch.from_numpy(faces), "L":new_laplacian}, f"{mesh_name}_refined_{num_iters}.pt")
verts = V.detach().cpu().numpy()
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts),
    o3d.utility.Vector3iVector(faces)
)