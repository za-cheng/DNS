import torch
import numpy as np
from torch.functional import norm
from CookTorranceRendering import apply_lighting_cook_torrance
from utils import sample_lights_from_equirectangular_image
def activation_2nd_ord(func, x, dx_dh=None, d2x_dh2=None):
    '''
    func must be element-wise activation
    x: ([n_batch], ndim)
    dx_dh: Jacobian, tensor of shape ([n_batch], ndim, nhdim)
    d2x_dh2: Diagonal of Hessian, tensor of shape  ([n_batch], ndim, nhdim)
    '''
    y, dy_dh, d2y_dh2 = func(x), None, None
    if dx_dh is not None:
        dy_dx = func.grad(x,ord=1).unsqueeze(dim=-1) #([n_batch], ndim, 1)
        dy_dh = dy_dx * dx_dh
        
        if d2x_dh2 is not None:
            d2y_dx2 = func.grad(x,ord=2).unsqueeze(dim=-1) #([n_batch], ndim, 1)
            d2y_dh2 = d2y_dx2 * (dx_dh**2) + d2x_dh2 * dy_dx # ([n_batch], ndim, nhdim)
        
    return y, dy_dh, d2y_dh2
        
def linear_2nd_ord(linear, x, dx_dh=None, d2x_dh2=None):
    '''
    linear must be linear layer
    x: ([n_batch], ndim)
    dx_dh: Jacobian, tensor of shape ([n_batch], ndim, nhdim)
    d2x_dh2: Diagonal of Hessian, tensor of shape  ([n_batch], ndim, nhdim)
    '''
    y, dy_dh, d2y_dh2 = linear(x), None, None
    if dx_dh is not None:
        dy_dh = linear.weight @ dx_dh
        if d2x_dh2 is not None:
            d2y_dh2 = linear.weight @ d2x_dh2
            
    return y, dy_dh, d2y_dh2

class ActivationFunc(object):
    def __init__(self):
        pass
    def __call__(self, x):
        return x
    def grad(self, x, ord):
        if ord == 1 or ord == 2:
            return torch.ones_like(x) * (2 - ord)
        raise NotImplementedError

class Sigmoid(ActivationFunc):
    def __call__(self, x):
        return torch.sigmoid(x)
    def grad(self, x, ord):
        y = torch.sigmoid(x)
        dy = y*(1-y)
        if ord == 1:
            return dy
        if ord == 2:
            return (1-2*y) * dy
        raise NotImplementedError

class Tanh(ActivationFunc):
    def __call__(self, x):
        return torch.tanh(x)
    def grad(self, x, ord):
        y = torch.tanh(x)
        dy = 1-y**2
        if ord == 1:
            return dy
        if ord == 2:
            return -2*y*dy
        raise NotImplementedError

class Sinusoidal(ActivationFunc):
    def __call__(self, x):
        return torch.sin(x)
    def grad(self, x, ord):
        if ord == 1:
            return torch.cos(x)
        if ord == 2:
            return -torch.sin(x)
        raise NotImplementedError
        
class LeakyReLU(ActivationFunc):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    def __call__(self, x):
        return torch.nn.functional.leaky_relu(x, self.negative_slope)
    def grad(self, x, ord):
        if ord == 1:
            return (x >= 0).to(x.dtype) * (1-self.negative_slope) + self.negative_slope
        if ord == 2:
            return torch.zeros_like(x)
        raise NotImplementedError

class Normalise(ActivationFunc):
    def __init__(self, eps=1e-12):
        super(Normalise, self).__init__()
        self.eps = eps
    def __call__(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)
    def grad(self, x, ord):
        raise NotImplementedError

### initialising SIREN ###

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-30 / num_input, 30 / num_input)

### end SIREN ###

class MLP(torch.nn.Module):
    def __init__(self, in_dim=0, h_dims=[], activations=[]):
        super(MLP, self).__init__()
        assert len(h_dims) == len(activations)
        self.layers = torch.nn.ModuleList()
        self.activations = list(map( {
            'none': ActivationFunc(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'relu': LeakyReLU(0),
            'leaky_relu': LeakyReLU(0.01),
            'lrelu': LeakyReLU(0.01),
            'sin': Sinusoidal(),
            'siren': Sinusoidal(),
        }.get, activations))
        
        self.in_dim = in_dim
        for i in range(len(h_dims)):
            out_dim = h_dims[i]
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            if activations[i] == 'siren':
                if i == 0:
                    first_layer_sine_init(self.layers[-1])
                else:
                    sine_init(self.layers[-1])
            in_dim = out_dim

        self.out_dim = (h_dims[-1] if len(h_dims)>0 else 0)
    
    def forward(self, x, dx_dh=None, d2x_dh2=None):
        for linear, activation in zip(self.layers, self.activations):
            x, dx_dh, d2x_dh2 = linear_2nd_ord(linear, x, dx_dh, d2x_dh2)
            x, dx_dh, d2x_dh2 = activation_2nd_ord(activation, x, dx_dh, d2x_dh2)
        return x, dx_dh, d2x_dh2


class PositionEncoding(MLP):
    def __init__(self, W, output_weight=None):
        '''
        W: transform matrix of shape (ndim_fourier_features, ndim_in)
        '''
        super(PositionEncoding, self).__init__()
        weight = torch.cat((W,W), dim=0)
        out_dim, in_dim = weight.shape
        bias = torch.zeros_like(weight[:,0])
        bias[out_dim//2:] += np.pi/2
        
        self.layers = torch.nn.ModuleList()
        self.activations = [Sinusoidal()]
        
        linear = torch.nn.Linear(in_dim, out_dim, True)
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        linear.bias = torch.nn.Parameter(bias, requires_grad=False)
        
        self.layers.append(linear)
        self.in_dim, self.out_dim = in_dim, out_dim
        if output_weight is not None:
            self.output_weight = torch.nn.Parameter(torch.cat((output_weight,output_weight), dim=0), requires_grad=False)
        else:
            self.output_weight = None
    def forward(self, *args):
        x, dx_dh, d2x_dh2 = super(PositionEncoding, self).forward(*args)
        if self.output_weight is None:
            return x, dx_dh, d2x_dh2
        else:
            x = x * self.output_weight
            if dx_dh is not None:
                dx_dh = dx_dh * self.output_weight.reshape(-1,1)
            if d2x_dh2 is not None:
                d2x_dh2 = d2x_dh2 * self.output_weight.reshape(-1,1)
            return x, dx_dh, d2x_dh2


        
class ResNet(MLP):
    def __init__(self, h_dims, activations):
        super(ResNet, self).__init__(h_dims[-1], h_dims, activations)
    def forward(self, x, dx_dh=None, d2x_dh2=None):
        y, dy_dh, d2y_dy2 = super(ResNet, self).forward(x, dx_dh, d2x_dh2)
        y = y + x
        if dy_dh is not None:
            dy_dh = dy_dh + dx_dh
        if d2y_dy2 is not None:
            d2y_dy2 = d2y_dy2 + d2x_dh2
        return y, dy_dh, d2y_dy2

class Sequential(MLP):
    def __init__(self, *mlps):
        super(Sequential, self).__init__()
        self.mlps = torch.nn.ModuleList(mlps)
        self.in_dim, self.out_dim = mlps[0].in_dim, mlps[-1].out_dim
    def forward(self, x, dx_dh=None, d2x_dh2=None):
        for mlp in self.mlps:
            x, dx_dh, d2x_dh2 = mlp(x, dx_dh, d2x_dh2)
        return x, dx_dh, d2x_dh2


class ShapeNet(torch.nn.Module):
    def __init__(self, velocity_mlp, T):
        super(ShapeNet, self).__init__()
        # assert velocity_mlp.in_dim == velocity_mlp.out_dim
        self.T = T
        self.velocity_mlp = velocity_mlp
    def forward(self, x, compute_derivative=0, compute_normals=False, source_normals=None):
        '''
        x must be of shape ([n_batch], ndim)
        returns all points along entire trajectory, and the velocities and their Jacobian and 
        '''
        # assert len(x.shape) <= 2 and x.shape[-1] == self.velocity_mlp.in_dim
        assert compute_derivative in (0,1,2)
        
        ndim = x.shape[-1]
        
        x_arr, v_arr, dv_dh_arr, d2v_d2h_arr = [x], [], [], []
        
        dx_dh = torch.eye(ndim, dtype=x.dtype, device=x.device)
        d2x_dh2 = torch.zeros(ndim,ndim, dtype=x.dtype, device=x.device)
        
        if compute_derivative < 1:
            dx_dh = None
        if compute_derivative < 2:
            d2x_dh2 = None
            
        for _ in range(self.T):
            v, dv_dh, d2v_d2h = self.velocity_mlp(x, dx_dh, d2x_dh2)
            x = x + v / self.T * 2
            
            x_arr.append(x)
            v_arr.append(v)
            dv_dh_arr.append(dv_dh)
            d2v_d2h_arr.append(d2v_d2h)
            
            
            if compute_derivative >= 1:
                dx_dh = dx_dh + dv_dh / self.T * 2
            if compute_derivative >= 2:
                d2x_dh2 = d2x_dh2 + d2v_d2h / self.T * 2
        
        if compute_normals:
            if source_normals is None:
                source_normals = x # source is sphere by default
            normals = (torch.inverse(dx_dh) @ source_normals.reshape(-1,3,1)).reshape(-1,3)
            normals = torch.nn.functional.normalize(normals, dim=-1)
            return x_arr, v_arr, dv_dh_arr, d2v_d2h_arr, normals

        return x_arr, v_arr, dv_dh_arr, d2v_d2h_arr
    
    def velocity_at(self, x):
        return self.velocity_mlp(x)[0]


class BRDFNet(torch.nn.Module):
    def __init__(self, brdf_mlp, constant_fresnel=True):
        '''
        TODO: specular color is single-channeled, may change this to RGB for metallic BRDFs
        '''
        super(BRDFNet, self).__init__()
        assert brdf_mlp.out_dim % 3 == 0 and brdf_mlp.out_dim >= 6
        self.brdf_mlp = brdf_mlp
        self.n_lobes = brdf_mlp.out_dim // 3 - 1
        self.constant_fresnel = constant_fresnel
    def _activation(self):
        def acti_func(x):
            n_lobes = self.n_lobes
            diffuse = torch.nn.functional.relu(x[...,:3])
            roughness = torch.sigmoid(x[...,3:3+n_lobes])
            specular = torch.nn.functional.relu(x[...,3+n_lobes:3+n_lobes*2])
            r0 = torch.nn.functional.sigmoid(x[...,3+n_lobes*2:3+n_lobes*3])
            if self.constant_fresnel:
                r0 = torch.ones_like(r0)
            return torch.cat((
                            diffuse, 
                            roughness, 
                            specular,
                            r0, #torch.ones_like(r0)
                            ), dim=-1)
        return acti_func   

    def forward(self, x):
        return self._activation()(self.brdf_mlp(x)[0])

class NormalNet(torch.nn.Module):
    def __init__(self, normal_mlp):
        super(NormalNet, self).__init__()
        self.normal_mlp = normal_mlp
    def forward(self, x):
        return torch.nn.functional.normalize(self.normal_mlp(x)[0], dim=-1)


class AmbientReflectionNet(torch.nn.Module):
    def __init__(self, diffuse_mlp, specular_mlp):
        super().__init__()
        self.diffuse_mlp = diffuse_mlp
        self.specular_mlp = specular_mlp
    def forward(self, normals, view_dirs, roughness, r0):
        assert roughness.shape[-1] == 1 and r0.shape[-1] == 1
        # TODO: add support to multi lobes

        normals = torch.nn.functional.normalize(normals, dim=-1)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
        
        inputs_specular = torch.cat((normals, view_dirs, roughness, r0), dim=-1)

        visible = (normals * view_dirs).sum(dim=-1) > 0

        diffuse_visible_rgb = self.diffuse_mlp(normals[visible])[0]
        specular_visible_rgb = self.specular_mlp(inputs_specular[visible])[0]

        # diffuse_rgb = torch.where(visible, diffuse_rgb, torch.tensor([0.0], dtype=diffuse_rgb.dtype).to(visible.device))
        # specular_rgb = torch.where(visible, specular_rgb, torch.tensor([0.0], dtype=diffuse_rgb.dtype).to(visible.device))

        diffuse_rgb = torch.zeros_like(normals)
        specular_rgb = torch.zeros_like(normals)
        diffuse_rgb[visible] = diffuse_visible_rgb
        specular_rgb[visible] = specular_visible_rgb

        return diffuse_rgb, specular_rgb

    @torch.no_grad()
    def shader(self, points, normals, lights, cameras, materials, textures, environ_map_name='pano4s.png', environ_map_size=100):
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
        _, (ambient_lights_rads, ambient_light_dirs) = sample_lights_from_equirectangular_image(environ_map_name, (environ_map_size*2, environ_map_size), device=points.device, intensity_ratio=0.25)
        split_size = 100
        points = points.reshape(-1,3)
        normals = normals.reshape(-1,3)
        textures = textures.squeeze(0)

        view_dirs = cameras.get_camera_center().reshape(-1,1,3) - points # (N, P, 3)

        normals = torch.nn.functional.normalize(normals, dim=-1)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1).squeeze(0)
        forward_facing = (normals * view_dirs).sum(dim=-1) > 0

        diffuse_albedo = textures[...,0:3]
        n_lobes = (textures.shape[-1] - 3) // 3
        assert n_lobes*3+3 == textures.shape[-1]
        roughness = textures[...,3:3+n_lobes]
        specular_albedo = textures[...,3+n_lobes:3+2*n_lobes]
        r0 = textures[...,3+2*n_lobes:3+3*n_lobes]

        pbar = (list(zip(torch.split(ambient_lights_rads, split_size), torch.split(ambient_light_dirs, split_size))))
        diffuse_total, specular_total = 0, 0
        for light_rads, light_dirs in pbar:
            
            diffuse_ratio, specular_ratio, _ = apply_lighting_cook_torrance(normals, light_dirs, view_dirs, roughness, r0)
            diffuse = (diffuse_ratio.unsqueeze(dim=-1) * light_rads.unsqueeze(dim=1)).sum(dim=0)
            specular = (specular_ratio.unsqueeze(dim=-1) * light_rads.unsqueeze(dim=1)).sum(dim=0)

            diffuse_total += diffuse
            specular_total += specular
        
        diffuse_color = diffuse_albedo * diffuse_total.reshape(diffuse_albedo.shape)
        specular_color = specular_albedo * specular_total.reshape(diffuse_albedo.shape)

        colors = diffuse_color + specular_color
        
        return colors, forward_facing

    def shader_(self, points, normals, lights, cameras, materials, textures):
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
        view_dirs = cameras.get_camera_center().reshape(-1,1,3) - points # (N, P, 3)

        normals = torch.nn.functional.normalize(normals, dim=-1)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
        forward_facing = (normals * view_dirs).sum(dim=-1) > 0

        diffuse_albedo = textures[...,0:3]
        n_lobes = (textures.shape[-1] - 3) // 3
        assert n_lobes*3+3 == textures.shape[-1]
        roughness = textures[...,3:3+n_lobes]
        specular_albedo = textures[...,3+n_lobes:3+2*n_lobes]
        r0 = textures[...,3+2*n_lobes:3+3*n_lobes]

        diffuse, specular = self.__call__(normals.reshape(-1,3), view_dirs.reshape(-1,3), roughness.reshape(-1,n_lobes), r0.reshape(-1,n_lobes))

        diffuse_color = diffuse_albedo * diffuse.reshape(diffuse_albedo.shape)
        specular_color = specular_albedo * specular.reshape(diffuse_albedo.shape)

        colors = diffuse_color + specular_color
        
        return colors, forward_facing



