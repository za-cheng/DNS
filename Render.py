import numpy as np
import torch
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PointLights, SoftPhongShader, DirectionalLights
)

from CookTorranceRendering import HardCookTorranceShader, SoftCookTorranceShader, SoftNormalShader, SoftXYZShader, SoftTextureShader
from CustomRendering import SoftCustomShader, HardCustomShader
from MerlRendering import HardMerlShader, SoftMerlShader, HardMerlMultiLightsShader
from utils import r2R

def render_mesh(mesh, modes, rotations, translations, image_size, blur_radius, faces_per_pixel, 
                device, background_colors=None, light_poses=None, materials=None, camera_settings=None, verts_radiance=None, multi_lights=None, ambient_net=None, sigma=1e-4, gamma=1e-4):

    if isinstance(modes, str):
        return render_mesh(mesh, (modes,), rotations, translations, image_size, blur_radius, faces_per_pixel, 
                device, background_colors, light_poses, materials, camera_settings, verts_radiance, multi_lights, ambient_net, sigma, gamma)[0]

    if camera_settings is None:
        cameras = OpenGLPerspectiveCameras(device=device)
    else:
        cameras = PerspectiveCameras(device=device, **camera_settings)
    

    if background_colors is None:
        background_colors = [None] * len(modes)

    # rasterization
    raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=min(np.log(1. / 1e-6 - 1.) * sigma, blur_radius / image_size * 2), 
            faces_per_pixel=faces_per_pixel, 
            perspective_correct= False,
        )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # shading
    shaders = []
    for mode, bgc in zip(modes, background_colors):
        if mode == 'image_ct':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftCookTorranceShader(device=device, cameras=cameras, blend_params=blend_params)
        elif mode == 'texture':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftTextureShader(device=device, cameras=cameras, blend_params=blend_params)
        elif mode == 'image_merl':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = HardMerlShader(device=device, cameras=cameras, blend_params=blend_params, materials=materials)
        elif mode == 'image_merl_multi_lights':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = HardMerlMultiLightsShader(device=device, cameras=cameras, blend_params=blend_params, materials=materials, multi_lights=multi_lights)
        elif mode == 'silhouette':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftSilhouetteShader(blend_params=blend_params)
        elif mode == 'normal':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftNormalShader(device=device, cameras=cameras, blend_params=blend_params)
        elif mode == 'ambient':
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftCustomShader(device=device, cameras=cameras, blend_params=blend_params, shading_func=ambient_net.shader)
        elif mode in ('xyz', 'depth'):
            if bgc is None:
                bgc = (0,0,0)
            blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
            shader = SoftXYZShader(device=device, cameras=cameras, blend_params=blend_params)
        else:
            raise ValueError(f'unrecognised mode for rendering \'{mode}\'')

        shaders.append(shader)

    images = [list() for _ in modes]

    for i, (r, T) in enumerate(zip(rotations, translations)):
        if r.ndim == 1:
            R = r2R(r)
        else:
            R = r
        if light_poses is None:
            location = (-R@T)[None]
            lights = PointLights(device=device, location=location)
        else:
            location = light_poses[i][None]# @ R
            lights = DirectionalLights(device=device, direction=location)
        
        fragments = rasterizer(mesh, R=R[None], T=T[None])
        
        for j, (shader, mode) in enumerate(zip(shaders, modes)):

            if mode == 'image_merl_multi_lights':
                img = shader(fragments, mesh, R=R[None], T=T[None]) # do not use active light as lights are already supplied
            else:
                img = shader(fragments, mesh, R=R[None], T=T[None], lights=lights)

            if mode == 'silhouette':
                images[j].append(img[0,...,3:4])
            elif mode == 'depth':
                depth = (img[...,:3] * R[:,-1]).sum(-1,keepdim=True)+T[-1]
                depth[img[...,3] <= 0.01] = 0
                images[j].append(depth[0])
            elif mode == 'texture':
                images[j].append(img[0,...,:3])
            else:
                images[j].append(img[0,...,:3])
        
    images = tuple(torch.stack(imgs, dim=0) for imgs in images)
    
    return images


    

