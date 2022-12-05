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
)
from pytorch3d.renderer.lighting import PointLights, DirectionalLights

from pytorch3d.renderer.materials import Materials
import numpy as np

import h5py
from glob import glob

def dot(tensor1, tensor2, dim=-1, keepdim=False, non_negative=False, epsilon=1e-6) -> torch.Tensor:
    x =  (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)
    if non_negative:
        x = torch.clamp_min(x, epsilon)
    return x


def _cook_torrance_shading(normal_vecs, incident_vecs, view_vecs, roughness, r0=None, epsilon=1e-10):
    '''
    normal_vecs, incident_vecs, view_vecs: (...,3) normalised vectors
    roughness: (...,k_lobes) rms slope
    r0: (...,k_lobes) fresnel factor
    returns: (...,k_lobes) specular factors
    '''
    half_vecs = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    
    # Beckmann model for D
    h_n = dot(half_vecs, normal_vecs, non_negative=True)
    cos_alpha_sq = h_n**2 # (...)
    cos_alpha_sq = cos_alpha_sq.unsqueeze(dim=-1) # (..., 1)
    cos_alpha_r_sq = torch.clamp_min(cos_alpha_sq*(roughness**2), epsilon) # (..., k_lobes)
    # D = torch.exp( (cos_alpha_sq - 1) /  cos_alpha_r_sq ) / \
    #     ( np.pi * cos_alpha_r_sq * cos_alpha_sq ) # (..., k_lobes)  # Beckmann

    # GGX model
    roughness_sq = roughness**2
    D = roughness_sq / np.pi / (cos_alpha_sq * (roughness_sq-1) + 1)**2 # GGX
    


    # Geometric term G
    v_n = dot(incident_vecs, view_vecs, non_negative=True) # (...)
    v_h = dot(half_vecs, view_vecs, non_negative=True) # (...)
    i_n = dot(incident_vecs, normal_vecs, non_negative=True) # (...)
    G = torch.clamp_max(torch.min(i_n, v_n) * 2 * h_n / v_h, 1) # (...)
    G = G.unsqueeze(dim=-1) # (..., 1)

    # Schlick's approximation for F
    if r0 is None:
        F = 1
    else:
        F = r0 + (1-r0) * ((1 - v_h.unsqueeze(dim=-1)) ** 5) # (..., k_lobes)

    return (D*F*G) / (np.pi*i_n*v_n).unsqueeze(dim=-1) # (..., k_lobes)

def _apply_lighting_cook_torrance(
    points, normals, lights, cameras, materials, textures, eps=1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: should be None. not used in this function
        textuers: tensor of shape (N, P, kdim), where kdim is of form
                  (diffuse_rgb=3, [roughness=1 * n_lobes, specular_rgb=1 * n_lobes, r0=1 * n_lobes] ])

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    if isinstance(lights, DirectionalLights):
        light_dirs = lights.direction.reshape(-1, 1, 3) + torch.zeros_like(points)
    elif isinstance(lights, PointLights):
        light_dirs = lights.location.reshape(-1,1,3) - points # (N, P, 3)
    else:
        raise TypeError
    view_dirs = cameras.get_camera_center().reshape(-1,1,3) - points # (N, P, 3)

    normals_ = torch.nn.functional.normalize(normals, dim=-1)
    light_dirs_ = torch.nn.functional.normalize(light_dirs, dim=-1)
    view_dirs_ = torch.nn.functional.normalize(view_dirs, dim=-1)


    falloff = dot(normals_, light_dirs_)
    forward_facing = dot(normals, view_dirs) > eps
    visible_mask = ((falloff > 0) & forward_facing) # (N, P) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (N, P) cosine falloff, 0 if not visible

    diffuse_albedo = textures[...,0:3]
    n_lobes = (textures.shape[-1] - 3) // 3
    assert n_lobes*3+3 == textures.shape[-1]
    roughness = textures[...,3:3+n_lobes]
    specular_albedo = textures[...,3+n_lobes:3+2*n_lobes]
    r0 = textures[...,3+2*n_lobes:3+3*n_lobes]

    specular_reflectance = (_cook_torrance_shading(normals_, light_dirs_, view_dirs_, roughness, r0) * specular_albedo).sum(-1, keepdim=True)
    irradiance = torch.unsqueeze(falloff, dim=-1) * torch.unsqueeze(lights.diffuse_color, dim=1) 

    diffuse_color = diffuse_albedo * irradiance
    specular_color = specular_reflectance * irradiance
    if isinstance(lights, PointLights):
        diffuse_color = diffuse_color / torch.clamp(dot(light_dirs, light_dirs, keepdim=True), min=eps)
        specular_color = specular_color / torch.clamp(dot(light_dirs, light_dirs, keepdim=True), min=eps)

    # ambient_color = 0 #lights.ambient_color
    
    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )
    return diffuse_color, specular_color, forward_facing


def apply_lighting_cook_torrance(
    normals, light_dirs, view_dirs, roughness, r0=1, eps=1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        normals, view_dirs: torch tensor of shape (P, 3).
        lights_dirs: torch tensor of shape (L, 3).
        roughness, r0: torch tensor of shape (P, n_lobes).

    Returns:
        diffuse_reflectance: torch tensor of shape (L, P)
        specular_reflectance: torch tensor of shape (L, P)
    """
    
    light_dirs = light_dirs.unsqueeze(dim=1) # (L, 1, 3)

    normals_ = torch.nn.functional.normalize(normals.unsqueeze(dim=0), dim=-1) # (1, P, 3)
    light_dirs_ = torch.nn.functional.normalize(light_dirs, dim=-1) # (L, 1, 3)
    view_dirs_ = torch.nn.functional.normalize(view_dirs.unsqueeze(dim=0), dim=-1) # (1, P, 3)

    falloff = dot(normals_, light_dirs_) # (L, P)
    forward_facing = dot(normals, view_dirs) > eps # (1, P)
    visible_mask = ((falloff > 0) & forward_facing) # (L, P) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (L, P) cosine falloff, 0 if not visible

    specular_reflectance = _cook_torrance_shading(normals_, light_dirs_, view_dirs_, roughness, r0).sum(dim=-1) # (L, P)
    

    return falloff, specular_reflectance * falloff, forward_facing

def cook_torrance_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
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
    diffuse, specular, opacity = _apply_lighting_cook_torrance(
        pixel_coords, pixel_normals, lights, cameras, materials, texels,
    )
    colors = diffuse + specular
    colors = colors.reshape(i_shape[:-1] + (-1,))
    opacity = opacity.reshape(i_shape[:-1])
    return colors, opacity

def normal_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
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
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    i_shape = pixel_normals.shape
    pixel_normals = pixel_normals.reshape(i_shape[:-1] + (-1,))
    return pixel_normals, torch.ones_like(pixel_normals[...,0])

def xyz_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
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
    i_shape = pixel_coords.shape
    pixel_coords = pixel_coords.reshape(i_shape[:-1] + (-1,))
    return pixel_coords, torch.ones_like(pixel_coords[...,0])


def texture_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
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
    i_shape = pixel_coords.shape
    texels = texels.reshape(i_shape[:-1] + (-1,))
    print(texels.shape)
    return texels, torch.ones_like(pixel_coords[...,0])

class HardCookTorranceShader(nn.Module):
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
        colors, opacity = cook_torrance_shading(
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


class SoftCookTorranceShader(nn.Module):
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
        colors, opacity = cook_torrance_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_rgb_blend(colors, opacity.float(), fragments, blend_params)
        return images

class SoftNormalShader(nn.Module):
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

        # texels = meshes.sample_textures(fragments)
        # lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors, opacity = normal_shading(
            meshes=meshes,
            fragments=fragments,
            texels=None,
            lights=None,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_rgb_blend(colors, opacity.float(), fragments, blend_params)
        return images

class SoftXYZShader(nn.Module):
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

        texels = None
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors, opacity = xyz_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_rgb_blend(colors, opacity.float(), fragments, blend_params)
        return images

class SoftTextureShader(nn.Module):
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
        colors, opacity = texture_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_texture_blend(colors, opacity.float(), fragments, blend_params)
        return images

def softmax_rgb_blend(
    colors, opacity, fragments, blend_params, znear: float = 1.0, zfar: float = 100
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        opacity: (N, H, W, K) boolean variable whether this face should be used for blending.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma) * opacity

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[20]: Argument `max` expected.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors



def softmax_texture_blend(
    colors, opacity, fragments, blend_params, znear: float = 1.0, zfar: float = 100
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        opacity: (N, H, W, K) boolean variable whether this face should be used for blending.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, colors.shape[-1]), dtype=colors.dtype, device=colors.device)
    background = 0

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma) * opacity

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[20]: Argument `max` expected.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :] = (weighted_colors + weighted_background) / denom
    #pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors
# if __name__ == '__main__':
#     materials = load_merl_materials(n_angles=1)
#     basis, mean = compute_dictionary(materials, 20, log_scale=True)


    
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

    merl_material = None #load_merl_material('BRDF/red-specular-plastic.h5', n_angles=3, device=device)
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

    textures = torch.ones_like(verts[None,...,:1]).expand((1,) + verts.shape[:-1] + (4,))
    textures = textures + 0
    textures[...,-1] = 0.1
    textures=TexturesVertex(verts_features=textures)

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
        faces_per_pixel=8, 
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
            raster_settings=soft_raster_settings
        ),
        shader=SoftCookTorranceShader(device=device, cameras=cameras, lights=lights, materials=merl_material, blend_params=blend_params)
    )
    img = hard_phong_renderer(meshes_world=sph_mesh, R=R, T=T)

    import cv2
    img = img.detach().cpu().numpy()[...,2::-1] * (15)
    cv2.imwrite('rendered.png', np.clip(img*255, 0, 255).astype(np.uint8)[0])
