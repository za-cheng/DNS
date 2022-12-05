import torch
from pytorch3d.renderer.mesh import TexturesVertex
import os
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PointLights, SoftPhongShader
)

from CookTorranceRendering import HardCookTorranceShader, SoftCookTorranceShader, SoftNormalShader, SoftXYZShader

from MerlRendering import (
    HardMerlShader, 
    SoftMerlShader,
    load_merl_material,
    load_merl_materials,
    compute_dictionary,
)

from Render import render_mesh
from Meshes import Meshes
import numpy as np
from pytorch3d.io import load_obj
from utils import *
from Render import render_mesh
from tqdm import tqdm
import random
import trimesh

def return_mesh(path, pattern, brdf_strs, device):

    trg_obj = os.path.join(path)
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces = faces.verts_idx.to(device)
    verts = verts.to(device)

    if path == 'david.obj':
        verts = verts @ torch.tensor([[-1,0,0],
                                      [0,0,1],
                                      [0,1,0]],dtype=verts.dtype, device=verts.device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh

    #### ANTICHOKE
    if pattern == 'checkerboard':
        verts_rgb1 = torch.zeros_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb2 = torch.zeros_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb3 = torch.zeros_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb4 = torch.zeros_like(verts)[None,:,:1]  # (1, V, 1)
        verts_mask_x = ((verts[None,:,:1] // 0.2) % 2 == 0)
        verts_mask_y = ((verts[None,:,1:2] // 0.2) % 2 == 0)
        verts_rgb1[verts_mask_x & verts_mask_y] = 1
        verts_rgb2[verts_mask_x & (~verts_mask_y)] = 1
        verts_rgb3[(~verts_mask_x) & verts_mask_y] = 1
        verts_rgb4[(~verts_mask_x) & (~verts_mask_y)] = 1
        verts_rgb = torch.cat((verts_rgb1, verts_rgb2, verts_rgb3, verts_rgb4), -1)
        assert len(brdf_strs) == 4
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)

#     #### 
    
#     ###### READING
    elif pattern == 'gradient':
        verts_rgb1 = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb2 = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb3 = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_rgb = verts + 1
        verts_rgb = verts_rgb / verts_rgb.sum(-1,keepdim=True)
        
        assert len(brdf_strs) == 3
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)

#     ######
    
    ##### BUNNY
    elif pattern == 'stripe_x':
        verts_rgb = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_mask = verts[None,:,:1] % 0.3 > 0.15
        verts_rgb[verts_mask] = 0
        verts_rgb = torch.cat((verts_rgb, 1-verts_rgb), -1)
        assert len(brdf_strs) == 2
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)

    
    elif pattern == 'stripe_y':
        verts_rgb = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_mask = verts[None,:,1:2] % 0.3 > 0.15
        verts_rgb[verts_mask] = 0
        verts_rgb = torch.cat((verts_rgb, 1-verts_rgb), -1)
        assert len(brdf_strs) == 2
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)
    
    elif pattern == 'stripe_z':
        verts_rgb = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_mask = verts[None,:,2:3] % 0.3 > 0.15
        verts_rgb[verts_mask] = 0
        verts_rgb = torch.cat((verts_rgb, 1-verts_rgb), -1)
        assert len(brdf_strs) == 2
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)

    #####
    
    ###### 
    elif pattern == 'spots':
        cents = verts[torch.randperm(len(verts))[:1000]]  # (100, 3)
        from sklearn.neighbors import NearestNeighbors
        
        dis, ind = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cents.cpu().numpy()).kneighbors(verts.cpu().numpy())
        print(dis.shape)
        verts_rgb = torch.ones_like(verts)[None,:,:1]  # (1, V, 1)
        verts_mask = torch.from_numpy(dis<=0.03).bool().cuda().reshape(1,-1)
        verts_rgb[verts_mask] = 0
        verts_rgb = torch.cat((verts_rgb, 1-verts_rgb), -1)
        
        assert len(brdf_strs) == 2
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)

#     ######
    
    elif pattern is None:
        mesh = trimesh.load(trg_obj)
        verts, faces, verts_rgb = mesh.vertices.astype(np.float32), mesh.faces, mesh.visual.to_color().vertex_colors[...,:3].astype(np.float32)

        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces = torch.from_numpy(faces).to(device)
        verts = torch.from_numpy(verts).to(device)
        verts_rgb = torch.from_numpy(verts_rgb).to(device)
        verts_rgb = verts_rgb / 255
        verts_rest = 3-verts_rgb.sum(-1,keepdims=True)
        verts_rgb = torch.cat([verts_rgb, verts_rest], dim=-1)
        verts_rgb = verts_rgb * torch.tensor([2, 5, 3, 1], dtype=verts_rgb.dtype, device=verts_rgb.device)
        verts_rgb = verts_rgb / verts_rgb.sum(-1,keepdims=True)

        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        #assert len(brdf_strs) == 4
        gt_materials = load_merl_materials([f'BRDF/{brdf_str}.h5' for brdf_str in brdf_strs], device=device, n_angles=2)


    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    trg_mesh = Meshes(
        verts=[verts],   
        faces=[faces], 
        vert_textures=[verts_rgb.reshape(-1, verts_rgb.shape[-1])]
    )

    return trg_mesh, gt_materials

def sample_view_angels(n_sample, elevation_range, azimuth_range):
    minz, maxz = np.sin(elevation_range[0]*np.pi/180), np.sin(elevation_range[1]*np.pi/180)
    elevations = np.arcsin(np.clip(np.random.rand(n_sample) * (maxz-minz) + minz, -0.999, 0.999)) * 180 / np.pi
    azimuths = np.random.rand(n_sample) * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
    return elevations, azimuths

def random_RT(n_views, distance_range, device, elevation_range=(-89,89), azimuth_range=(0,360)):
    '''
    generates random camera poses on uniform sphere facing the centre at (0,0,0)
    '''
    elevations, azimuths = sample_view_angels(n_views, elevation_range, azimuth_range)
    
    dis_min, dis_max = distance_range
    distances = np.random.rand(n_views) * (dis_max - dis_min) + dis_min
    
    Rs, Ts = zip(*[look_at_view_transform(distance, elevation, azimuth, device=device) for distance, elevation, azimuth in zip(distances, elevations, azimuths)])
    Rs = torch.cat(Rs, dim=0)
    Ts = torch.cat(Ts, dim=0)
    rs = R2r(Rs)
    
    return rs, Ts


@torch.no_grad()
def synthesize_imgs(path, pattern, brdf_strs, r, t, **params):

    params = dotty(params)
    
    mesh, materials = return_mesh(path, pattern, brdf_strs, params['device'])

    images = render_mesh(mesh, 'image_merl', r, t, params['rendering.rgb.image_size'], #######
                        0.0, 1, params['device'], materials=materials,# background_colors=((1,1,1),),
                        sigma=1e-6, gamma=1e-6)
    
    silhouettes = render_mesh(mesh, 'silhouette', r, t, params['rendering.silhouette.image_size'],
                        0.0, 1, params['device'], materials=materials, 
                        sigma=1e-6, gamma=1e-6)                   
    
    silhouettes_bin_th = 0.01
    silhouettes[silhouettes < silhouettes_bin_th ] = 0
    silhouettes[silhouettes >= silhouettes_bin_th ] = 1

    return images, silhouettes

@torch.no_grad()
def synthesize_geometry_maps(path, pattern, brdf_strs, r, t, **params):

    params = dotty(params)

    mesh, materials = return_mesh(path, pattern, brdf_strs, params['device'])

    normal = render_mesh(mesh, 'normal', r, t, params['rendering.rgb.image_size'],
                        0.0, 1, params['device'], materials=materials, 
                        sigma=1e-6, gamma=1e-6)
    
    xyz = render_mesh(mesh, 'xyz', r, t, params['rendering.rgb.image_size'],
                        0.0, 1, params['device'], materials=materials, 
                        sigma=1e-6, gamma=1e-6)                   


    return normal, xyz

@torch.no_grad()
def synthesize_ambient_reflection_imgs(path, pattern, brdf_strs, r, t, env_light_map, env_light_size, intensity_ratio=1.0, **params):

    params = dotty(params)

    mesh, materials = return_mesh(path, pattern, brdf_strs, params['device'])

    if env_light_size is None:
        env_light_size = cv2.imread(env_light_map).shape[0]

    _, (light_radiance, light_dir) = sample_lights_from_equirectangular_image(env_light_map, (env_light_size*2, env_light_size), params['device'], intensity_ratio=intensity_ratio)

    n_samples_per_batch = 10
    
    env_images = 0
    for light_rads, light_dirs in \
            tqdm(list(zip(torch.split(light_radiance, n_samples_per_batch), torch.split(light_dir, n_samples_per_batch)))):
        environ_lights = DirectionalLights(diffuse_color=light_rads.reshape(-1,3), direction=light_dirs.reshape(-1,3), device=light_rads.device)
        _env_images = render_mesh(mesh, 'image_merl_multi_lights', r, t, params['rendering.rgb.image_size'],
                                    0.0, 1, params['device'], materials=materials, 
                                    sigma=1e-6, gamma=1e-6, multi_lights=environ_lights)

        env_images += _env_images

    return env_images





