
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

from CookTorranceRendering import softmax_rgb_blend, hard_rgb_blend

def custom_shading_callback(meshes, fragments, lights, cameras, materials, texels, shader_callback, mask=None) -> torch.Tensor:
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
    colors, opacity = shader_callback(
        pixel_coords, pixel_normals, lights, cameras, materials, texels,
    )
    colors = colors.reshape(i_shape[:-1] + (-1,))

    opacity = opacity.reshape(i_shape[:-1])
    return colors, opacity


class SoftCustomShader(nn.Module):
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
        self, device="cpu", cameras=None, blend_params=None, lights=None, materials=None, shading_func=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.shading_func = shading_func
        self.lights = lights
        self.materials = materials

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
        shading_func = kwargs.get("shading_func", self.shading_func)
        colors, opacity = custom_shading_callback(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
            shader_callback=self.shading_func,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = softmax_rgb_blend(colors, opacity.float(), fragments, blend_params)
        return images


class HardCustomShader(nn.Module):
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
        self, device="cpu", cameras=None, blend_params=None, lights=None, materials=None, shading_func=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.shading_func = shading_func
        self.lights = lights
        self.materials = materials

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
        shading_func = kwargs.get("shading_func", self.shading_func)
        colors, opacity = custom_shading_callback(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
            shader_callback=self.shading_func,
        )
        # if max_radiance is not None:
        #     colors = torch.clamp_max(colors, max_radiance)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images