from typing import List, Union

import torch

import pytorch3d
from pytorch3d.structures import utils as struct_utils
from pytorch3d.renderer.mesh import TexturesVertex

class Meshes(pytorch3d.structures.Meshes):
    def __init__(self, verts=None, faces=None, vert_textures=None, verts_normals=None):
        if vert_textures is not None:
            super(Meshes, self).__init__(verts=verts, faces=faces, textures=TexturesVertex(vert_textures))
        else:
            super(Meshes, self).__init__(verts=verts, faces=faces)
        if verts_normals is not None:
            self._verts_normals_packed = torch.nn.functional.normalize(
                struct_utils.list_to_packed(verts_normals)[0], eps=1e-6, dim=1
            )
