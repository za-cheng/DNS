import torch
import numpy as np
from skimage.measure import marching_cubes as marching_cubes_lewiner
import trimesh


def xor_3d_gird(grid):
    '''
    grid: nd boolean tensor where n>3
    returns: xor nd boolean tensor of same shape
        where xor[...,i,j,k]=True only if mask[...,i,j,k] and at least one of its 6 neighbors xor True
    '''
    grid_p = torch.nn.functional.pad(grid, (1,1,1,1,1,1), "constant", False)
    xor = (grid_p[...,0:-2,1:-1,1:-1] ^ grid) | (grid_p[...,2:,1:-1,1:-1] ^ grid) | \
          (grid_p[...,1:-1,0:-2,1:-1] ^ grid) | (grid_p[...,1:-1,2:,1:-1] ^ grid) | \
          (grid_p[...,1:-1,1:-1,0:-2] ^ grid) | (grid_p[...,1:-1,1:-1,2:] ^ grid) 
    return xor

def project_to_pixel_coords(world_coords, proj_mat):
    '''
    world_coords: [n,3]
    proj_mat: [3,4]
    returns: [n,2] pixel coordinates
    '''

    wc_homogeneous = torch.cat((world_coords, torch.ones_like(world_coords[...,:1])), dim=-1) # [n,4]
    pc = wc_homogeneous @ proj_mat.T # [n,3]
    return pc[...,:-1] / pc[...,-1:]

def carve_siloutte(grid, grid_coords, proj_mat, silhouette_img, retain_out_of_image_voxels=False, silhouette_th=0.5):
    wc = grid_coords[grid]
    pc = project_to_pixel_coords(wc, proj_mat)
    pc_round = torch.round(pc).to(torch.int64)
    
    h, w = silhouette_img.shape[:2]
    in_view_mask = (pc_round[...,0] >= 0) & (pc_round[...,0] < w) & (pc_round[...,1] >= 0) & (pc_round[...,1] < h)

    if retain_out_of_image_voxels:
        mask = torch.ones_like(in_view_mask)
    else:
        mask = in_view_mask

    query_pc = pc_round[in_view_mask]
    mask[in_view_mask] = (silhouette_img[query_pc[:,1], query_pc[:,0]] >= silhouette_th).reshape(-1)

    grid = grid.clone()
    grid[grid] = mask
    return grid

def get_visual_hull_grid(grid_size, grid_span, proj_mats, silhouette_imgs, retain_out_of_image_voxels=False, silhouette_th=0.5):
    '''
    grid_size: (nx, ny, nz), size of grid in n voxels
    grid_span: (xmin, xmax, ymin, ymax, zmin, zmax) bouding box in world units
    ''' 
    nx, ny, nz = grid_size
    xmin, xmax, ymin, ymax, zmin, zmax = grid_span

    x_space = (xmax-xmin) / nx
    y_space = (ymax-ymin) / ny
    z_space = (zmax-zmin) / nz

    device = proj_mats.device
    grid = torch.ones(nx, ny, nz, dtype=torch.bool, device=device)
    grid_coords = torch.empty(nx, ny, nz, 3, dtype=proj_mats.dtype, device=device)

    grid_coords[...,0] = xmin - x_space/2 + x_space * torch.arange(nx, device=device).reshape(-1,1,1)
    grid_coords[...,1] = ymin - y_space/2 + y_space * torch.arange(ny, device=device).reshape(1,-1,1)
    grid_coords[...,2] = zmin - z_space/2 + z_space * torch.arange(nz, device=device).reshape(1,1,-1)

    for proj_mat, silhouette_img in zip(proj_mats, silhouette_imgs):
        grid = carve_siloutte(grid, grid_coords, proj_mat, silhouette_img, retain_out_of_image_voxels, silhouette_th)

    return grid, grid_coords


@torch.no_grad()
def get_visual_hull_mesh(grid_size, grid_span, proj_mats, silhouette_imgs, retain_out_of_image_voxels=False, silhouette_th=0.5, smooth=True, max_faces=None):

    nx, ny, nz = grid_size
    xmin, xmax, ymin, ymax, zmin, zmax = grid_span

    x_space = (xmax-xmin) / nx
    y_space = (ymax-ymin) / ny
    z_space = (zmax-zmin) / nz

    grid, grid_coords = get_visual_hull_grid(grid_size, grid_span, proj_mats, silhouette_imgs, retain_out_of_image_voxels, silhouette_th)

    verts, faces, normals, values = marching_cubes_lewiner(1-grid.cpu().numpy())

    verts = verts * [x_space, y_space, z_space] + [xmin - x_space/2, ymin - y_space/2, zmin - z_space/2]

    if smooth or (max_faces is not None):
        mesh = trimesh.Trimesh(verts, faces)
        if smooth:
            trimesh.smoothing.filter_laplacian(mesh)
        if max_faces is not None:
            mesh = mesh.simplify_quadratic_decimation(max_faces)
        verts, faces = mesh.vertices, mesh.faces

    device = grid.device
    dtype = grid_coords.dtype

    return torch.tensor(verts, device=device, dtype=dtype), torch.tensor(faces, device=device, dtype=torch.int64)




