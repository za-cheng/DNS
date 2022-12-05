import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import pytorch3d
import torch
import numpy as np
from Synthetic import synthesize_imgs, random_RT
from Loss import velocity_loss, clipped_mae, chamfer_3d
from torch.nn.functional import mse_loss as mse
from Render import render_mesh
from Model import MLP, PositionEncoding, ResNet, Sequential, ShapeNet, BRDFNet

from utils import manual_seed, rand_ico_sphere, save_models, load_models
from Meshes import Meshes
from tqdm import tqdm
import h5py
import trimesh
from Model import MLP, PositionEncoding, Sequential, ShapeNet, BRDFNet
from itertools import chain
from utils import dotty, sample_lights_from_equirectangular_image, save_images, random_dirs, P_matrix_to_rot_trans_vectors, pytorch_camera, compile_video
import cv2
import os
from diligent import diligent_eval_chamfer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

@torch.no_grad()
def sample_mesh(shape_net, brdf_net, init_mesh=None, normal_net=None, **params):
    params = dotty(params)
    if init_mesh is None:
        init_mesh = rand_ico_sphere(params['sampling.ico_sphere_level'], device=device)
            
    s = init_mesh.verts_packed()

    x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
    theta_x = brdf_net(s)

    x = x_arr[-1]

    faces = init_mesh.faces_packed()
    mesh = Meshes(verts=[x], faces=[faces], vert_textures=[theta_x])

    return mesh

def clamp_vertex_grad(grad, thre):
    ret = grad + 0
    ret[torch.logical_not(torch.abs(ret) < thre)] = 0
    return ret

def train(images, silhouettes, rotations, translations, shape_net, brdf_net, optimizer, n_iterations, light_dirs=None, call_back=None, init_mesh=None,  camera_settings=None, camera_settings_silhoutte=None,  **params):
    
    device = params['device']
    n_images = len(images)
    params = dotty(params)

    null_init = init_mesh is None

    def closure():
        #################################
        ## sample mesh from neural nets
        nonlocal init_mesh
        if null_init:
            init_mesh = rand_ico_sphere(params['sampling.ico_sphere_level'], device=device)
        
        s = init_mesh.verts_packed()

        if (params['loss.lambda_velocity']==0) or (params['loss.alpha'] == 0) or params['training.compute_velocity_seperately']:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
        else:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 2)

        x = x_arr[-1]

        if params.get('training.vertex_grad_clip', None) is not None:
            hook = x.register_hook(lambda grad: clamp_vertex_grad(grad, params['training.vertex_grad_clip']))

        theta_x = brdf_net(s)

        faces = init_mesh.faces_packed()


        mesh = Meshes(verts=[x], faces=[faces], vert_textures=[theta_x])

        #################################
        ## render images with mesh
        ## and compute losses
        batch_idx = torch.randperm(n_images)[:params['training.n_image_per_batch']]
        loss_image, loss_silhouette, loss_velocity = 0, 0, 0

        for i in batch_idx:

            gt_image, gt_silhouette = images[i:i+1], silhouettes[i:i+1]

            light_pose = None
            if light_dirs is not None:
                light_pose = light_dirs[i:i+1]
            if params['loss.lambda_image'] != 0:
                prd_image = render_mesh(mesh, 
                        modes='image_ct', #######
                        rotations=rotations[i:i+1], 
                        translations=translations[i:i+1], 
                        image_size=params['rendering.rgb.image_size'], 
                        blur_radius=params['rendering.rgb.blur_radius'], 
                        faces_per_pixel=params['rendering.rgb.faces_per_pixel'], 
                        device=device, background_colors=None, light_poses=light_pose, materials=None, camera_settings=camera_settings,
                        sigma=params['rendering.rgb.sigma'], gamma=params['rendering.rgb.gamma'])
               

                max_intensity = params['rendering.rgb.max_intensity'] #* (np.random.rand()+1)

                loss_tmp = clipped_mae(gt_image.cuda().clamp_max(max_intensity), prd_image, max_intensity) / params['training.n_image_per_batch']
                
                (loss_tmp * params['loss.lambda_image']).backward(retain_graph=True)
                loss_image += loss_tmp.detach()

            if params['loss.lambda_silhouette'] != 0:
                prd_silhouette = render_mesh(mesh, 
                        modes='silhouette', 
                        rotations=rotations[i:i+1], 
                        translations=translations[i:i+1], 
                        image_size=params['rendering.silhouette.image_size'], 
                        blur_radius=params['rendering.silhouette.blur_radius'], 
                        faces_per_pixel=params['rendering.silhouette.faces_per_pixel'], 
                        device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings_silhoutte,
                        sigma=params['rendering.silhouette.sigma'], gamma=params['rendering.silhouette.gamma'])

                loss_tmp = mse(gt_silhouette.cuda(), prd_silhouette[...,0]) / params['training.n_image_per_batch']
                (loss_tmp * params['loss.lambda_silhouette']).backward(retain_graph=True)
                loss_silhouette += loss_tmp.detach()

        if params['loss.lambda_velocity'] == 0:
            pass
        elif (params['loss.alpha'] == 0) or (not params['training.compute_velocity_seperately']):
            loss_tmp = velocity_loss(v_arr, d2v_d2s_arr, params['loss.alpha'])
            (loss_tmp * params['loss.lambda_velocity']).backward(retain_graph=True)
            loss_velocity = loss_tmp.detach()
        else:
            ico_sphere = rand_ico_sphere(params['training.sampling_lvl_for_vel_loss'], device=device)
            if init_mesh is not None:
                n_verts = ico_sphere.verts_packed().shape[0]
                s = pytorch3d.ops.sample_points_from_meshes(init_mesh, n_verts).reshape(n_verts, 3)
            else:
                s = ico_sphere.verts_packed()
            n_pts_total = s.shape[0]
            for _s in torch.split(s, params['training.n_pts_per_split'], dim=0):
                n_pts = _s.shape[0]
                _x_arr, _v_arr, _dv_ds_arr, _d2v_d2s_arr = shape_net(_s, 2)
                loss_tmp = velocity_loss(_v_arr, _d2v_d2s_arr, params['loss.alpha']) * n_pts / n_pts_total
                (loss_tmp * params['loss.lambda_velocity']).backward(retain_graph=False)
                loss_velocity += loss_tmp.detach()

        if params['loss.lambda_edge'] != 0:
            loss_edge = pytorch3d.loss.mesh_edge_loss(mesh)
            (loss_edge * params['loss.lambda_edge']).backward(retain_graph=True)
        else:
            loss_edge = 0

        if params['loss.lambda_normal_consistency'] != 0:
            loss_normal_consistency = pytorch3d.loss.mesh_normal_consistency(mesh)
            (loss_normal_consistency * params['loss.lambda_normal_consistency']).backward(retain_graph=True)
        else:
            loss_normal_consistency = 0

        if params['loss.lambda_laplacian_smoothing'] != 0:
            loss_laplacian_smoothing = pytorch3d.loss.mesh_laplacian_smoothing(mesh)
            (loss_laplacian_smoothing * params['loss.lambda_laplacian_smoothing']).backward(retain_graph=True)
        else:
            loss_laplacian_smoothing = 0


        loss =  loss_image * params['loss.lambda_image'] + \
                loss_silhouette * params['loss.lambda_silhouette'] + \
                loss_velocity * params['loss.lambda_velocity']  + \
                loss_edge * params['loss.lambda_edge'] + \
                loss_normal_consistency * params['loss.lambda_normal_consistency']+ \
                loss_laplacian_smoothing * params['loss.lambda_laplacian_smoothing']

        return mesh.detach(), (float(loss), float(loss_image), float(loss_silhouette), float(loss_velocity), float(loss_edge), float(loss_normal_consistency), float(loss_laplacian_smoothing))
    
    pbar = tqdm(range(n_iterations))

    for N_IT in pbar:
        optimizer.zero_grad()
        mesh, losses = closure()

        optimizer.step()
        pbar.set_description('|'.join(f'{l:.2e}' for l in losses).replace('e', '').replace('|', ' || ', 1))

        if call_back is not None:
            call_back(mesh, losses[0])

    call_back(end=True)
    return losses

def call_back(mesh=None, loss=None, end=False):

    if end:
        call_back.history = []
        call_back.video_writer.release()
        call_back.video_writer = None
        return

    if not hasattr(call_back, 'history'):
        call_back.history = []
    call_back.history.append(float(loss))

    if min(call_back.history) == loss:
        save_models(f'{checkpoint_name}', brdf_net=brdf_net, shape_net=shape_net, 
                    optimizer=optimizer, meta=dict(loss=loss, params=dict(params)))
    
    with torch.no_grad():
        frame = render_mesh(mesh, 
                            modes='image_ct', #######
                            rotations=r[[0,9,40]], 
                            translations=t[[0,9,40]], 
                            image_size=params['rendering.rgb.image_size'], 
                            blur_radius=params['rendering.rgb.blur_radius'], 
                            faces_per_pixel=params['rendering.rgb.faces_per_pixel'], 
                            device=device, background_colors=None, light_poses=lights[[0,9,40]], materials=None, camera_settings=camera_settings)
        
        frame = torch.cat(list(f for f in frame), dim=1)
        frame = np.clip((frame * 255 / params['rendering.rgb.max_intensity']).cpu().numpy(), 0, 255).astype(np.uint8) #######
    if not hasattr(call_back, 'video_writer') or call_back.video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        call_back.video_writer = cv2.VideoWriter(f'{checkpoint_name}.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))
    
    call_back.video_writer.write((frame * np.ones((3), np.uint8))[...,::-1])



if __name__ == '__main__':

    params = dotty({
        'device': torch.device('cuda:0'),
        'n_lobes': 5,
        'training': 
        {
            'n_image_per_batch': 48,
            'lr': 1e-3,
            'compute_velocity_seperately': True,
            'n_pts_per_split': 2048,
            'sampling_lvl_for_vel_loss': 5,
            'n_iterations': 4500, #######
            'rand_seed': 0,
            'vertex_grad_clip': 0.1,
        },
        'sampling':
        {
            'ico_sphere_level': 6, #######
            'T': 20,
        },
        'rendering':
        {
            'rgb': 
            {
                'image_size': 512, #######
                'blur_radius': 0.0, #######
                'faces_per_pixel': 4, #######
                'max_intensity': 0.15, #######   read_ing:0.09, budd_ha: 0.15, pot_2: 0.15, co_w: 0.15, bea_r:  0.2
                'sigma': 1e-4, #######
                'gamma': 1e-4, #######
            },
            'silhouette':
            {
                'image_size': 512, #######
                'blur_radius': 0.1, #######
                'faces_per_pixel': 100,
                'sigma': 1e-4,
                'gamma': 1e-4,
            },
        },
        'loss':
        {
            'lambda_image': 4.0,
            'lambda_silhouette': 1.0, #######
            'lambda_velocity': 0.1,# 0.1,
            'alpha': 0.01,# 0.05,
            'lambda_edge': 0.5,
            'lambda_normal_consistency': 0.5,
            'lambda_laplacian_smoothing': 0.5,
        }, 
    })

    device = params['device']
    manual_seed(params['training.rand_seed'])
    checkpoint_name = 'diligent_reading' #######  'diligent_reading'
    
    from diligent import load_diligent_mv_full
    images, silhouettes, lights, K, P, transf = load_diligent_mv_full('DiLiGenT-MV', 'reading')

    images = images.cpu()

    r, t = P_matrix_to_rot_trans_vectors(P)


    pos_encode_weight = torch.cat(tuple(torch.eye(3) * (1.5**i) for i in range(0,14,1)), dim=0) #######
    pos_encode_out_weight = torch.cat(tuple( torch.tensor([1.0/(1.3**i)]*3) for i in range(0,14,1)), dim=0) #######
    
    shape_net = ShapeNet(velocity_mlp= Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),
                        MLP(pos_encode_weight.shape[0]*2, [256,256,256,3], ['lrelu','lrelu','lrelu','tanh']),  
                        ), T=params['sampling.T']).to(device)


    brdf_net = BRDFNet( Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),  
                        MLP(pos_encode_weight.shape[0]*2, [256]*5+[params['n_lobes']*3+3], ['lrelu']*5+['none']),    
                        ), constant_fresnel=True).to(device)

    optimizer = torch.optim.Adam(list(shape_net.parameters())+list(brdf_net.parameters()), lr=params['training.lr'])
    camera_settings = pytorch_camera(params['rendering.rgb.image_size'], K)
    camera_settings_silhoutte = pytorch_camera(params['rendering.silhouette.image_size'], K)
    
    train(images, silhouettes, r, t, shape_net, brdf_net, optimizer, params['training.n_iterations'],  
            call_back=call_back, 
            light_dirs=lights,
            camera_settings = camera_settings,
            camera_settings_silhoutte=camera_settings_silhoutte,
            **params)

    load_models(f'{checkpoint_name}', brdf_net=brdf_net, shape_net=shape_net, 
                    optimizer=optimizer)
    
    mesh = sample_mesh(shape_net, brdf_net, **params)#init_mesh=init_mesh, **params)
    trimesh.Trimesh( ( mesh.verts_packed().detach() @ transf[:3,:3].T + transf[:3,-1]).cpu().numpy(), mesh.faces_packed().cpu().numpy()).export(f'{checkpoint_name}.obj')

    compile_video(mesh, f'{checkpoint_name}.mp4', distance=2, render_mode='image_ct', **params)