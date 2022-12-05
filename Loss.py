import torch
import pytorch3d
#import kornia
from torch.nn.functional import mse_loss as mse
import pytorch3d
import pytorch3d.loss

def velocity_loss(v_arr, d2v_dh2_arr, alpha=0.01):
    '''
    compute regularization term
    |[I-alpha*nabla^2] v|^2
    where nabla^2 is the diagonal Laplacian operator
    and I identity operator
    '''
    loss = 0
    T = 0
    
    for v, d2v_dh2 in zip(v_arr, d2v_dh2_arr):
        laplace_v = 0
        if alpha != 0:
            laplace_v = d2v_dh2.sum(-1)
        loss = loss + ( (v - alpha * laplace_v)**2 ).mean()
        T = T + 1
        
    return loss / T

def clipped_huber(x, y, max_val=1, beta=0.01):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = y-x
    diff = torch.where(mask, torch.zeros_like(diff), diff).abs()
    diff = torch.where(diff<=beta, (diff**2), 2*beta*diff.abs()-beta**2 )
    return diff.mean()

def clipped_mse(x, y, max_val=1, edge_lambda=None):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = (x - y)**2
    diff = torch.where(mask, torch.zeros_like(diff), diff)
    loss = diff.mean()
    if edge_lambda is not None and edge_lambda != 0:
        edge_mse = mse( kornia.filters.spatial_gradient(x.clamp_max(1).permute(0,3,1,2)), 
                        kornia.filters.spatial_gradient(y.clamp_max(1).permute(0,3,1,2)) )
        loss = loss + edge_lambda * edge_mse
    return loss



def clipped_mae(x, y, max_val=1):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = (x - y).abs()
    diff = torch.where(mask, torch.zeros_like(diff), diff)
    return diff.mean()

def clipped_shadow(x, y, max_val=1, min_val=0.1):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = y-x
    diff = torch.where(mask, torch.zeros_like(diff), diff)
    y = diff.abs().clamp_max(min_val)
    diff = torch.where((x>=min_val), (diff**2), 2*y*diff.abs()-y**2 )
    return diff.mean()


from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

def chamfer_3d(x, y, z_tresh=0):

    '''
    mean distance for every point in x to its nearest point in y
    '''
    x = x.reshape(1,-1,3)
    y = y.reshape(1,-1,3)

    if z_tresh > 0:
        x_mask = x[...,-1] > y[...,-1].min() * (1-z_tresh) + z_tresh * y[...,-1].max()
        y_mask = y[...,-1] > y[...,-1].min() * (1-z_tresh) + z_tresh * y[...,-1].max()

    x_nn = knn_points(x[x_mask].reshape(1,-1,3), y, lengths1=None, lengths2=None, K=1)
    y_nn = knn_points(y[y_mask].reshape(1,-1,3), x, lengths1=None, lengths2=None, K=1)

    cham_x = torch.sqrt(x_nn.dists[..., 0])  # (N, P1)
    cham_y = torch.sqrt(y_nn.dists[..., 0])  # (N, P1)
    

    # Apply point reduction
    cham_x = cham_x.mean(1)  # (N,)
    cham_y = cham_y.mean(1)  # (N,)
    return ( cham_x + cham_y ) * 0.5

def squared_point_mesh_distance(meshes, pcls):

    '''
    This function fails when faces are too small
    in which case a precision issue occurs and distances are wrong

    this seems to be fixed by pytorch3d github commit #88f5d79
    but i'm suspending this function in favor of chamfer_3d

    '''
    raise NotImplementedError

    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = pytorch3d.loss.point_mesh_distance.point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    return point_to_face


def mesh2mesh_chamfer(mesh1, mesh2, n_samples=1000000):
    pcd1 = pytorch3d.structures.Pointclouds([pytorch3d.ops.sample_points_from_meshes(mesh1, n_samples).reshape((n_samples,3))])
    pcd2 = pytorch3d.structures.Pointclouds([pytorch3d.ops.sample_points_from_meshes(mesh2, n_samples).reshape((n_samples,3))])

    pcd2_to_mesh1 = torch.sqrt(squared_point_mesh_distance(mesh1, pcd2)).mean()
    pcd1_to_mesh2 = torch.sqrt(squared_point_mesh_distance(mesh2, pcd1)).mean()

    return ( pcd1_to_mesh2 + pcd2_to_mesh1 ) * 0.5

def mesh2mesh_hausdorff(mesh1, mesh2, n_samples=1000000):
    pcd1 = pytorch3d.structures.Pointclouds([pytorch3d.ops.sample_points_from_meshes(mesh1, n_samples).reshape((n_samples,3))])
    pcd2 = pytorch3d.structures.Pointclouds([pytorch3d.ops.sample_points_from_meshes(mesh2, n_samples).reshape((n_samples,3))])

    pcd2_to_mesh1 = torch.sqrt(squared_point_mesh_distance(mesh1, pcd2))
    pcd1_to_mesh2 = torch.sqrt(squared_point_mesh_distance(mesh2, pcd1))

    return max( pcd1_to_mesh2.max(), pcd2_to_mesh1.max() )