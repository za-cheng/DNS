from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import scipy
import scipy.io
import cv2
import trimesh
from utils import dotty, P_matrix_to_rot_trans_vectors, pytorch_camera
from Render import render_mesh
from Meshes import Meshes
from utils import save_images

def make_torch_tensor(*args):
    return tuple(torch.tensor(a).cuda() for a in args)

def load_diligent_mv(path_str, name_str, view_id):
    path = f'{path_str}/mvpmsData/{name_str}PNG'

    lights_ints = np.loadtxt(f'{path}/view_{view_id:02}/light_intensities.txt').astype(np.float32)
    light_dirs = np.loadtxt(f'{path}/view_{view_id:02}/light_directions.txt')#@np.diag([-1,1,1])
    light_dirs = light_dirs.astype(np.float32)
    view_dirs = (np.zeros_like(light_dirs) + [0,0,1]).astype(np.float32)
    
    imgs = np.stack([\
                     cv2.imread(f'{path}/view_{view_id:02}/{j+1:03}.png', -1)[...,::-1].astype(np.float32) \
                for j in range(len(lights_ints))], axis=0)
    imgs = imgs / lights_ints.reshape(lights_ints.shape[0], 1, 1, lights_ints.shape[1]) / 65535

    mask = cv2.imread(f'{path}/view_{view_id:02}/mask.png', -1) > 100
    normal = scipy.io.loadmat(f'{path}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32)
    R = scipy.io.loadmat(f'{path}/Calib_Results.mat')[f'Rc_{view_id}']
    T = scipy.io.loadmat(f'{path}/Calib_Results.mat')[f'Tc_{view_id}']
    K = np.zeros((3,4), dtype=np.float32)
    K[:3,:3] = scipy.io.loadmat(f'{path}/Calib_Results.mat')['KK']

    # R_correct = np.diag([1.003458, 1.0, 1.0]) # because R in Diligent dataset is not exactly orthogonal
    # R = np.linalg.inv(R_correct) @ R # ad hoc correction
    # K[:3,:3] = K[:3,:3] @ R_correct # ad hoc correction
    
    P = np.eye(4, dtype=np.float32)
    P[:3,:3] = R
    P[:3,3] = T.reshape(-1)

    # make images square
    K[0,2] -= 50
    imgs = imgs[:,:,50:-50]
    mask = mask[:,50:-50]
    normal = normal[:,50:-50]
    
    imgs[:,~mask] = 0

    colocated_mask = (light_dirs[...,-1] > 0.65)

    return make_torch_tensor(imgs.astype(np.float32)[colocated_mask], 
        mask.astype(np.float32), (light_dirs[colocated_mask]@np.diag([1,-1,-1])@R).astype(np.float32), view_dirs[colocated_mask], normal, K, P)

def load_diligent_mesh(path_str, name_str):
    path = f'{path_str}/mvpmsData/{name_str}PNG'
    mesh_path = f'{path_str}/mvpmsData/{name_str}PNG/mesh_Gt.ply'

    mesh = trimesh.load(mesh_path)
    verts, faces = make_torch_tensor(mesh.vertices.astype(np.float32), mesh.faces)
    shift = verts.mean(0)
    scale = (verts - shift).abs().max()
    transf = torch.eye(4).cuda()
    transf[:3,:3] = torch.eye(3).cuda() * scale
    transf[:3,3] = shift

    return (verts - shift) / scale, faces, transf



def load_diligent_mv_full(path_str, name_str, max_faces_no=None):
    images, masks, lights, Ps = [], [], [], []
    for view_id in range(1,21):
        imgs, mask, light_dirs, view_dirs, _, K, P = load_diligent_mv(path_str, name_str, view_id)
        n_imgs = imgs.shape[0]
        images.append(imgs)
        masks.append(mask.expand((n_imgs,) + mask.shape))
        lights.append(light_dirs)
        Ps.append(P.expand((n_imgs,) + P.shape))
    gt_verts, gt_faces, transf = load_diligent_mesh(path_str, name_str)
    scale = transf[0,0]
    Ps = torch.cat(Ps,dim=0) @ transf
    Ps[:,:3] = Ps[:,:3] / scale
    return torch.cat(images,dim=0), torch.cat(masks,dim=0), torch.cat(lights,dim=0), K, Ps, transf

@torch.no_grad()
def diligent_eval(verts, faces, gt_verts, gt_faces, path, transf, K, Ps, masks=None, mode='normal', transform_verts=False, normals=None):
    
    r, t = P_matrix_to_rot_trans_vectors(Ps)
    camera_settings = pytorch_camera(512, K)
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]
        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)
        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
            if normals is not None:
                normals = normals @ torch.inverse(R).transpose(-1,-2)
            

    estimates = render_mesh(Meshes(verts=[verts], faces=[faces], verts_normals=([normals] if normals is not None else None)), 
                        modes=mode, #######
                        rotations=r, 
                        translations=t, 
                        image_size=512, 
                        blur_radius=0.00, 
                        faces_per_pixel=4, 
                        device=verts.device, background_colors=None, camera_settings=camera_settings,
                        sigma=1e-4, gamma=1e-4)

    ground_truths = render_mesh(Meshes(verts=[gt_verts], faces=[gt_faces]), 
                        modes=mode, #######
                        rotations=r, 
                        translations=t, 
                        image_size=512, 
                        blur_radius=0, 
                        faces_per_pixel=8, 
                        device=verts.device, background_colors=None, camera_settings=camera_settings,
                        sigma=1e-4, gamma=1e-4)

    

    if mode == 'depth':
        estimates = estimates.unsqueeze(dim=-1)
        ground_truths = ground_truths.unsqueeze(dim=-1)
    elif mode == 'normal':
        shape = estimates.shape
        estimates = torch.nn.functional.normalize((estimates.reshape(shape[0], -1, 3)@Ps[:,:3,:3].transpose(-1,-2)).reshape(shape), dim=-1)
        estimates = estimates * torch.tensor([1,-1,-1], dtype=estimates.dtype, device=estimates.device)
        ground_truths = np.stack([scipy.io.loadmat(f'{path}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32) for view_id in range(1,21)], 0)
        ground_truths = torch.from_numpy(ground_truths).cuda()[:,:,50:-50]
        ground_truths = torch.nn.functional.normalize(ground_truths, dim=-1)
        save_images((estimates+1)/2, 'est', 1)
        save_images((ground_truths+1)/2, 'gt', 1)

    fg_masks = (torch.norm(estimates, dim=-1) > 0) & (torch.norm(ground_truths, dim=-1) > 0)
    if masks is not None:
        fg_masks = fg_masks & masks
    
    if mode == 'normal':
        error = torch.acos((estimates * ground_truths).sum(-1).clamp(-1,1)) * 180 / np.pi
    elif mode == 'depth':
        error = (estimates - ground_truths).abs() * scale

    error[~fg_masks] = 0

    return error, fg_masks


def eval_normal(path_str, name_str, method):
    path = f'{path_str}/mvpmsData/{name_str}PNG'
    ground_truths = np.stack([scipy.io.loadmat(f'{path}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32) for view_id in range(1,21)], 0)
    
    if method == 'li':
        path = f'{path_str}/estNormal/{name_str}PNG_Normal_TIP19Li_view'
    elif method == 'park':
        path = f'{path_str}/estNormal/{name_str}PNG_Normal_PAMI16Park_view'
    estimates = np.stack([scipy.io.loadmat(f'{path}{view_id}.mat')['Normal_est'].astype(np.float32) for view_id in range(1,21)], 0)


    fg_masks = (np.linalg.norm(estimates, axis=-1) > 0) & (np.linalg.norm(ground_truths, axis=-1) > 0)

    error = np.arccos((estimates * ground_truths).sum(-1).clip(-1,1)) * 180 / np.pi
        
    error[~fg_masks] = 0

    return error, fg_masks

import pytorch3d
from Loss import chamfer_3d

@torch.no_grad()
def diligent_eval_chamfer(verts, faces, gt_verts, gt_faces, transf, transform_verts=False, n_points=1000000, z_thresh=0):
    
   
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]
        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)
        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
            

    mesh_est = Meshes(verts=[verts], faces=[faces])
    mesh_gt = Meshes(verts=[gt_verts], faces=[gt_faces])

    pcd_est = pytorch3d.ops.sample_points_from_meshes(mesh_est, n_points).reshape((-1,n_points,3))
    pcd_gt = pytorch3d.ops.sample_points_from_meshes(mesh_gt, n_points).reshape((-1,n_points,3))

    return chamfer_3d(pcd_est, pcd_gt, z_thresh) * scale


from Loss import mesh2mesh_chamfer
@torch.no_grad()
def diligent_eval_mesh_to_mesh(verts, faces, gt_verts, gt_faces, transf, transform_verts=False, n_points=1000000):
    
   
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]
        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)
        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
            

    mesh_est = Meshes(verts=[verts], faces=[faces])
    mesh_gt = Meshes(verts=[gt_verts], faces=[gt_faces])

    return mesh2mesh_chamfer(mesh_est, mesh_gt, n_points) * scale