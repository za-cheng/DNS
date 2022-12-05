# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Interface to pytorch3d 2.5.0

from typing import Tuple
import torch
from pytorch3d.ops import interpolate_face_attributes

import warnings

import torch
import torch.nn as nn
import math
from pytorch3d.renderer.blending  import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights, DirectionalLights

from pytorch3d.renderer.materials import Materials
import numpy as np

import h5py
from glob import glob

def dot(tensor1, tensor2, dim=-1, keepdim=False) -> torch.Tensor:
    return (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)

def _directions_to_rusink_angles(normal_vecs, incident_vecs, view_vecs, return_dim=3) -> torch.Tensor:
    """
    Args:
        normal_vecs, incident_vecs, view_vecs: tensors of shape (..., 3)
            inputs must be normalized vectors
        return_dim: integer in [1,3]
    Returns:
        angles: tensor of shape (N, return_dim) each of (theta_h, theta_d, phi_d) in radians
    """
    h = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    theta_h = torch.acos(torch.clamp((h*normal_vecs).sum(-1, keepdim=True), min=-1+1e-7, max=1-1e-7)) # (N, 1)
    if return_dim == 1:
        return theta_h
    theta_d = torch.acos(torch.clamp((incident_vecs*view_vecs).sum(dim=-1, keepdim=True), min=-1+1e-7, max=1-1e-7)) / 2
    if return_dim == 2:
        return torch.cat((theta_h, theta_d), dim=-1)
    hn = torch.nn.functional.normalize(torch.cross(h, normal_vecs), dim=-1)
    wh = torch.nn.functional.normalize(torch.cross(incident_vecs, view_vecs), dim=-1)
    hn_wh = dot(hn, wh)
    cos_phi_d = torch.where(dot(torch.cross(hn, wh), h) >= 0, hn_wh, -hn_wh)
    phi_d = torch.acos(torch.clamp(cos_phi_d, min=-1+1e-7, max=1-1e-7))
    phi_d = torch.unsqueeze(phi_d, dim=-1)
    return torch.cat((theta_h, theta_d, phi_d), dim=-1)


def _interp_materials(materials, angles) -> torch.Tensor:
    """
    Args:
        materials: torch tensor of shape (N, n_basis, n_color, n_theta_h, [n_theta_d, [n_phi_d]])
        indices: torch tensor of shape (N, P, k) or (P, k) where k can be (1,2,3) 
            is the number of angles in radians of (theta_h, [theta_d, [phi_d]])
            and must be in range
    
    Returns:
        interp_materials: tensor of shape (N, P, n_basis, n_color) where C is number
            of color channels
    """
    if angles.dim() == 2:
        angles = torch.unsqueeze(angles, dim=0)
    n_basis, n_channels = materials.shape[1:3]
    materials = materials.reshape(materials.shape[:1] + (-1,) + materials.shape[3:]) # (N, C, ...)
    N1, P, k1 = angles.shape
    N2, C = materials.shape[:2]
    k2 = materials.dim() - 2
    k = min(k1, k2)
    angles = angles[...,:k]
    if k2 > k:
        if k == 1 and k2 == 3:
            materials = materials[...,0,90]
        elif k == 1 and k2 == 2:
            materials = materials[...,0]
        elif k == 2 and k2 == 3:
            materials = materials[...,90]
        else:
            raise ValueError
    N = max(N1, N2)
    angles = angles / torch.tensor([math.pi/180*89, math.pi/180*89, math.pi/180*179], device=angles.device)[:k]
    if k == 1:
        angles = (angles * materials.shape[-1]).reshape(N1,1,P).expand(N,C,P)
        indices_floor = torch.clamp(torch.floor(angles).long(), 0, materials.shape[-1]-1)
        indices_ceil = torch.clamp(torch.ceil(angles).long(), 0, materials.shape[-1]-1)
        offsets = angles - indices_floor
        materials = materials.expand((N,) + materials.shape[1:])
        val_floor = torch.gather(materials, dim=-1, index=indices_floor) #(N, C, P)
        val_ceil = torch.gather(materials, dim=-1, index=indices_ceil) #(N, C, P)
        val = val_floor + (val_ceil-val_floor) * offsets #(N, C, P)
        val = val.transpose(1, 2) #(N, P, C)
        return val.reshape(val.shape[:2] + (n_basis, n_channels))
    if k == 2 or k == 3:
        angles = angles.flip(dims=[-1])
        angles = (angles*2-1).expand(N, P, k).view((N,) + (1,)*(k-1) + (P, k)) # (N, P, k)
        materials = materials.expand((N, C,) + materials.shape[2:]) # (N, C, k1, k2 ... )
        val = torch.nn.functional.grid_sample(materials, angles, mode='bilinear', padding_mode='zeros', align_corners=True) # (N, C, 1, 1..., P)
        val = val.reshape(N, C, P) 
        val = val.transpose(1, 2) #(N, P, C)
        return val.reshape(val.shape[:2] + (n_basis, n_channels))

def _apply_lighting(
    points, normals, lights, cameras, materials, textures, brdf_nd=2, eps=1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (L, P, 3) or (P, 3).
        normals: torch tensor of shape (L, P, 3) or (P, 3)
        lights: instance of the Lights class of shape (L, 3).
        cameras: instance of the Cameras class of (3,).
        materials: tensor of shape (N, P, no_basis, n_channel=3).
        textuers: tensor of shape (N, P, no_basis)

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    if isinstance(lights, DirectionalLights):
        light_dirs = lights.direction.reshape(-1, 1, 3) + torch.zeros_like(points) # (L, P, 3)
    elif isinstance(lights, PointLights):
        light_dirs = lights.location.reshape(-1,1,3) - points # (L, P, 3)
    else:
        raise TypeError
    view_dirs = cameras.get_camera_center().reshape(1,1,3) - points # (L, P, 3)

    normals_ = torch.nn.functional.normalize(normals, dim=-1)
    light_dirs_ = torch.nn.functional.normalize(light_dirs, dim=-1)
    view_dirs_ = torch.nn.functional.normalize(view_dirs, dim=-1)

    runsink_angles = _directions_to_rusink_angles(normals_, light_dirs_, view_dirs_, brdf_nd) # (L, P, brdf_nd)

    falloff = dot(normals_, light_dirs_)
    visible_mask = ((falloff > 0) & (dot(normals, view_dirs) > 0)) # (L, P) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (L, P) cosine falloff, 0 if not visible

    reflectance_basis = _interp_materials(materials, runsink_angles) # (L, P, no_basis, n_channels)
    # print(reflectance_basis.shape)
    reflectance = dot(reflectance_basis, torch.unsqueeze(textures, dim=-1), dim=2) # (N, P, color_channels=3)
    
    shading_color = reflectance * torch.unsqueeze(falloff, dim=-1) * torch.unsqueeze(lights.diffuse_color, dim=1) 
    if isinstance(lights, PointLights):
        shading_color = shading_color / torch.clamp(dot(light_dirs, light_dirs, keepdim=True), min=eps)

    ambient_color = 0 #lights.ambient_color
    
    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            shading_color.squeeze(),
        )
    return ambient_color, shading_color


def merl_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, no_basis)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    i_shape = pixel_coords.shape
    pixel_coords = pixel_coords.reshape(i_shape[0], -1, 3)
    pixel_normals = pixel_normals.reshape(i_shape[0], -1, 3)
    # texels = texels[...,0:1,:].expand(texels.shape) # use obly the nearest face's texture for blending
    texels = texels.reshape(pixel_normals.shape[:-1]+(-1,))
    ambient, reflected = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials, texels,
    )
    colors = ambient + reflected
    colors = colors.reshape(i_shape[:-1] + (-1,))
    return colors

class HardMerlShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        max_radiance = kwargs.get("max_radiance", None)
        colors = merl_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


class SoftMerlShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        max_radiance = kwargs.get("max_radiance", None)
        colors = merl_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


from itertools import product

def load_merl_material(binary_path, n_angles=3, device=None):
    '''
    load materials in a tensor
    '''
    assert n_angles >= 1 and n_angles <= 3
    with h5py.File(binary_path, 'r') as f:
        ret =  np.array(f['BRDF']) # (RGB=3, theta_h=90, theta_d=90, phi_d=180)
    if n_angles == 1:
        ret = ret[:,:,0,90]
    elif n_angles == 2:
        ret = ret[:,:,:,90]
    return torch.from_numpy(ret[None, None]).to(device)

def load_merl_materials(binary_paths=None, n_angles=3, device=None):
    if binary_paths is None:
        binary_paths = sorted(glob('BRDF/*.h5'))
    materials = [load_merl_material(path, n_angles, device) for path in binary_paths]
    return torch.cat(materials, dim=1).float()
    
def compute_dictionary(materials, k_basis=15, log_scale=False, log_eps=1e-20):
    # data_dims = materials.shape[2:]
    # materials = materials.reshape(materials.shape[:2] + (-1,)) # (N_batch, N_materials, N_channels*N_angles)

    data_dims = materials.shape[3:]
    materials = materials.reshape(materials.shape[0], materials.shape[1]*materials.shape[2], -1) # (N_batch, N_materials*N_channels, N_angles)

    if log_scale:
        materials = torch.log(torch.clamp_min(materials, log_eps))
    mean = materials.mean(dim=1) # (N_batches, N_channels*N_angles)
    materials = materials - materials.mean(dim=1, keepdims=True)
    # U, S, V = torch.svd(materials)
    U, S, V = np.linalg.svd(materials.cpu().numpy(), full_matrices=False)
    U = torch.from_numpy(U).to(materials.device)
    S = torch.from_numpy(S).to(materials.device)
    V = torch.from_numpy(V).to(materials.device).transpose(-1,-2)

    basis = V[:,:,:k_basis] * S[:,None,:k_basis] # (N_batches, N_channels*N_angles, k_basis)
    err = materials - U[:,:,:k_basis] @ basis.transpose(-1,-2)
    return basis.reshape((basis.shape[0],) + data_dims + (k_basis,)), mean.reshape((mean.shape[0],) + data_dims)


# if __name__ == '__main__':
#     materials = load_merl_materials(n_angles=1)
#     basis, mean = compute_dictionary(materials, 20, log_scale=True)



def merl_shading_multi_lights(meshes, fragments, multi_lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, no_basis)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    i_shape = pixel_coords.shape
    pixel_coords = pixel_coords.reshape(i_shape[0], -1, 3)
    pixel_normals = pixel_normals.reshape(i_shape[0], -1, 3)
    # texels = texels[...,0:1,:].expand(texels.shape) # use obly the nearest face's texture for blending
    texels = texels.reshape(pixel_normals.shape[:-1]+(-1,))
    ambient, reflected = _apply_lighting(
        pixel_coords, pixel_normals, multi_lights, cameras, materials, texels,
    )
    colors = (ambient + reflected).sum(dim=0, keepdims=True)
    colors = colors.reshape(i_shape[:-1] + (-1,))
    return colors


class HardMerlMultiLightsShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, multi_lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.multi_lights = multi_lights if multi_lights is not None else [PointLights(device=device)]
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        multi_lights = kwargs.get("multi_lights", self.multi_lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        max_radiance = kwargs.get("max_radiance", None)
        colors = merl_shading_multi_lights(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            multi_lights=multi_lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images

    
if __name__ == '__main__':
    from pytorch3d.io import load_obj, save_obj
    from pytorch3d.renderer.mesh import TexturesVertex
    from Meshes import Meshes
    from pytorch3d.transforms import Rotate, Translate
    from pytorch3d.utils import ico_sphere
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import (
        mesh_edge_loss, 
        mesh_laplacian_smoothing, 
        mesh_normal_consistency,
    )
    from chamfer import chamfer_distance
    import numpy as np
    from tqdm import tqdm
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib as mpl



    import imageio
    import torch.nn as nn
    import torch.nn.functional as F
    from skimage import img_as_ubyte

    from pytorch3d.renderer import (
        OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, PointLights
    )
    device = torch.device('cuda:0')
    # merl_material = load_merl_material('merl_brdfs/alum-bronze.binary', device='cuda:0')
    # np.save('merl_ab.npy', merl_material)

    merl_material = load_merl_material('BRDF/red-specular-plastic.h5', n_angles=3, device=device)
    # merl_material = torch.from_numpy(np.load('merl_ab.npy')).to(device)


    trg_obj = 'bunny.obj'
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    faces = faces.verts_idx.to(device)
    verts = verts.to(device)

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    textures=TexturesVertex(verts_features=torch.ones_like(verts[None,...,:1]))

    #sph_mesh = ico_sphere(5, device)
    sph_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Initialize an OpenGL perspective camera.
    cameras = OpenGLPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0))

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0, 
        faces_per_pixel=10, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )


    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    hard_raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    soft_raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma * .01, 
        faces_per_pixel=3, 
    )






    # class HardPhongShader(nn.Module):
    #     """
    #     Per pixel lighting - the lighting model is applied using the interpolated
    #     coordinates and normals for each pixel. The blending function hard assigns
    #     the color of the closest face for each pixel.

    #     To use the default values, simply initialize the shader with the desired
    #     device e.g.

    #     .. code-block::

    #         shader = HardPhongShader(device=torch.device("cuda:0"))
    #     """

    #     def __init__(
    #         self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    #     ):
    #         super().__init__()
    #         self.lights = lights if lights is not None else PointLights(device=device)
    #         self.materials = (
    #             materials if materials is not None else Materials(device=device)
    #         )
    #         self.cameras = cameras
    #         self.blend_params = blend_params if blend_params is not None else BlendParams()

    #     def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
    #         cameras = kwargs.get("cameras", self.cameras)
    #         if cameras is None:
    #             msg = "Cameras must be specified either at initialization \
    #                 or in the forward pass of HardPhongShader"
    #             raise ValueError(msg)

    #         texels = meshes.sample_textures(fragments)
    #         lights = kwargs.get("lights", self.lights)
    #         materials = kwargs.get("materials", self.materials)
    #         blend_params = kwargs.get("blend_params", self.blend_params)
    #         colors = phong_shading(
    #             meshes=meshes,
    #             fragments=fragments,
    #             texels=texels,
    #             lights=lights,
    #             cameras=cameras,
    #             materials=materials,
    #         )
    #         images = hard_rgb_blend(colors, fragments, blend_params)
    #         return images

    # def _apply_lighting(
    #     points, normals, lights, cameras, materials
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Args:
    #         points: torch tensor of shape (N, P, 3) or (P, 3).
    #         normals: torch tensor of shape (N, P, 3) or (P, 3)
    #         lights: instance of the Lights class.
    #         cameras: instance of the Cameras class.
    #         materials: instance of the Materials class.

    #     Returns:
    #         ambient_color: same shape as materials.ambient_color
    #         diffuse_color: same shape as the input points
    #         specular_color: same shape as the input points
    #     """
    #     light_diffuse = lights.diffuse(normals=normals, points=points)
    #     light_specular = lights.specular(
    #         normals=normals,
    #         points=points,
    #         camera_position=cameras.get_camera_center(),
    #         shininess=materials.shininess,
    #     )
    #     ambient_color = materials.ambient_color * lights.ambient_color
    #     diffuse_color = materials.diffuse_color * light_diffuse
    #     specular_color = materials.specular_color * light_specular
    #     if normals.dim() == 2 and points.dim() == 2:
    #         # If given packed inputs remove batch dim in output.
    #         return (
    #             ambient_color.squeeze(),
    #             diffuse_color.squeeze(),
    #             specular_color.squeeze(),
    #         )
    #     return ambient_color, diffuse_color, specular_color


    # def phong_shading(
    #     meshes, fragments, lights, cameras, materials, texels
    # ) -> torch.Tensor:
    #     """
    #     Apply per pixel shading. First interpolate the vertex normals and
    #     vertex coordinates using the barycentric coordinates to get the position
    #     and normal at each pixel. Then compute the illumination for each pixel.
    #     The pixel color is obtained by multiplying the pixel textures by the ambient
    #     and diffuse illumination and adding the specular component.

    #     Args:
    #         meshes: Batch of meshes
    #         fragments: Fragments named tuple with the outputs of rasterization
    #         lights: Lights class containing a batch of lights
    #         cameras: Cameras class containing a batch of cameras
    #         materials: Materials class containing a batch of material properties
    #         texels: texture per pixel of shape (N, H, W, K, 3)

    #     Returns:
    #         colors: (N, H, W, K, 3)
    #     """
    #     verts = meshes.verts_packed()  # (V, 3)
    #     faces = meshes.faces_packed()  # (F, 3)
    #     vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    #     faces_verts = verts[faces]
    #     faces_normals = vertex_normals[faces]
    #     pixel_coords = interpolate_face_attributes(
    #         fragments.pix_to_face, fragments.bary_coords, faces_verts
    #     )
    #     pixel_normals = interpolate_face_attributes(
    #         fragments.pix_to_face, fragments.bary_coords, faces_normals
    #     )
    #     ambient, diffuse, specular = _apply_lighting(
    #         pixel_coords, pixel_normals, lights, cameras, materials
    #     )
    #     colors = (ambient + diffuse) * texels + specular
    #     return colors




    R,T = look_at_view_transform(3, 0, 0, device=device)
    location = (-R[0]@T[0])[None]
    lights = PointLights(device=device, location=location)
    hard_phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=hard_raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, materials=merl_material, blend_params=blend_params)
    )
    img = hard_phong_renderer(meshes_world=sph_mesh, R=R, T=T)

    import cv2
    img = img.detach().cpu().numpy()[...,2::-1] * (1/0.003)
    cv2.imwrite('rendered.png', np.clip(img*255, 0, 255).astype(np.uint8)[0])
