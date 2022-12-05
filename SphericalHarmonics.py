import numpy as np
import torch

from numpy import pi
from math import sqrt


def rsh(l, m, coords, method=None):
    '''
    computes real sheprical harmonics
    l, m: integer, order and degree of sh basis
    coords: either [..., 2] representing (theta, phi) of zenith and azimuth angles in radians,
            or [..., 3] representing cartesian in (x,y,z), in which case it will be normalized
    delegates to hardcoded methods when l<=4, which are faster than naive implementation
    '''

    assert l >= 0 & m**2 <= l**2
    assert coords.shape[-1] == 2 or coords.shape[-1] == 3
    
    if method=='hard' or (method is None and l <= 4):
        if coords.shape[-1] == 2:
            cos_theta = torch.cos(coords[...,0])
            sin_theta = torch.sin(coords[...,0])
            cos_phi = torch.cos(coords[...,1])
            sin_phi = torch.sin(coords[...,1])
            cart = torch.stack((cos_phi*sin_theta, sin_phi*sin_theta, cos_theta), dim=-1)
        else:
            cart = torch.nn.functional.normalize(coords, dim=-1)
        return Hardcoded_RSH(l, m, cart)
    
    else:
        if coords.shape[-1] == 2:
            theta = coords[...,0]
            phi = coords[...,1]
        else:
            coords = torch.nn.functional.normalize(coords, dim=-1).clamp(-1,1)
            theta = torch.acos(coords[...,2])
            phi = torch.atan2(coords[...,1], coords[...,0])
        return Runtime_RSH(l, m, theta, phi)

def sq_factorial(*args):
    ret = 1.0
    for n in range(*args):
        ret *= sqrt(n+1)
    return ret

def doubleFactorial(x, cache=[1, 1, 2, 3, 8, 15, 48, 105, 384, 945, 3840, 10395, 46080, 135135, 645120, 2027025]):

  if x < len(cache):
    return cache[x] * 1.0
  else:
    if x % 2 == 0:
        s = cache[-2] * 1.0
    else:
        s = cache[-1] * 1.0
    n = x * 1.0
    while n >= len(cache):
      s *= n
      n -= 2.0
    return s

def LegendrePolynomial(l, m, x):
    pmm = 1.0
    if m > 0:
        pmm = doubleFactorial(2 * m - 1) * torch.pow(1 - x * x, m / 2.0)
        if m % 2 == 1:
            pmm = -pmm

    if l == m:
        return pmm

    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmm1

    for n in range(m + 2, l+1):
        pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m)
        pmm = pmm1
        pmm1 = pmn
    return pmm1

def Runtime_RSH(l, m, theta, phi):
    kml = sqrt((2 * l + 1) / (4 * pi )) / sq_factorial(l - abs(m), l + abs(m))
    if m > 0:
        return sqrt(2.0) * kml * torch.cos(m * phi) * LegendrePolynomial(l, m, torch.cos(theta))
    elif m < 0:
        return sqrt(2.0) * kml * torch.sin(-m * phi) * LegendrePolynomial(l, -m, torch.cos(theta))
    else:
        return kml * LegendrePolynomial(l, m, torch.cos(theta))


def Hardcoded_RSH(l, m, cart):
    if m == 0:
        m_str = '0'
    elif m > 0:
        m_str = f'p{m}'
    else:
        m_str = f'n{-m}'
    ufunc = globals()[f'HardcodedSH{l}{m_str}']
    
    return ufunc(cart)


def HardcodedSH00(cart) :
    # 0.5 * sqrt(1/pi)
    return 0.282095 + cart[...,0]*0


def HardcodedSH1n1(cart) :
    # -sqrt(3/(4pi)) * y
    return -0.488603 * cart[...,1]


def HardcodedSH10(cart) :
    # sqrt(3/(4pi)) * z
    return 0.488603 * cart[...,2]


def HardcodedSH1p1(cart) :
    # -sqrt(3/(4pi)) * x
    return -0.488603 * cart[...,0]


def HardcodedSH2n2(cart) :
    # 0.5 * sqrt(15/pi) * x * y
    return 1.092548 * cart[...,0] * cart[...,1]


def HardcodedSH2n1(cart) :
    # -0.5 * sqrt(15/pi) * y * z
    return -1.092548 * cart[...,1] * cart[...,2]


def HardcodedSH20(cart) :
    # 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
    return 0.315392 * (-cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1] + 2.0 * cart[...,2] * cart[...,2])


def HardcodedSH2p1(cart) :
    # -0.5 * sqrt(15/pi) * x * z
    return -1.092548 * cart[...,0] * cart[...,2]


def HardcodedSH2p2(cart) :
    # 0.25 * sqrt(15/pi) * (x^2 - y^2)
    return 0.546274 * (cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1])


def HardcodedSH3n3(cart) :
    # -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
    return -0.590044 * cart[...,1] * (3.0 * cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1])


def HardcodedSH3n2(cart) :
    # 0.5 * sqrt(105/pi) * x * y * z
    return 2.890611 * cart[...,0] * cart[...,1] * cart[...,2]


def HardcodedSH3n1(cart) :
    # -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
    return -0.457046 * cart[...,1] * (4.0 * cart[...,2] * cart[...,2] - cart[...,0] * cart[...,0]
                             - cart[...,1] * cart[...,1])


def HardcodedSH30(cart) :
    # 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
    return 0.373176 * cart[...,2] * (2.0 * cart[...,2] * cart[...,2] - 3.0 * cart[...,0] * cart[...,0]
                             - 3.0 * cart[...,1] * cart[...,1])


def HardcodedSH3p1(cart) :
    # -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
    return -0.457046 * cart[...,0] * (4.0 * cart[...,2] * cart[...,2] - cart[...,0] * cart[...,0]
                             - cart[...,1] * cart[...,1])


def HardcodedSH3p2(cart) :
    # 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
    return 1.445306 * cart[...,2] * (cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1])


def HardcodedSH3p3(cart) :
    # -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
    return -0.590044 * cart[...,0] * (cart[...,0] * cart[...,0] - 3.0 * cart[...,1] * cart[...,1])


def HardcodedSH4n4(cart) :
    # 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
    return 2.503343 * cart[...,0] * cart[...,1] * (cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1])


def HardcodedSH4n3(cart) :
    # -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
    return -1.770131 * cart[...,1] * cart[...,2] * (3.0 * cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1])


def HardcodedSH4n2(cart) :
    # 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
    return 0.946175 * cart[...,0] * cart[...,1] * (7.0 * cart[...,2] * cart[...,2] - 1.0)


def HardcodedSH4n1(cart) :
    # -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
    return -0.669047 * cart[...,1] * cart[...,2] * (7.0 * cart[...,2] * cart[...,2] - 3.0)


def HardcodedSH40(cart) :
    # 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
    z2 = cart[...,2] * cart[...,2]
    return 0.105786 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0)


def HardcodedSH4p1(cart) :
    # -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
    return -0.669047 * cart[...,0] * cart[...,2] * (7.0 * cart[...,2] * cart[...,2] - 3.0)


def HardcodedSH4p2(cart) :
  # 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
    return 0.473087 * (cart[...,0] * cart[...,0] - cart[...,1] * cart[...,1]) * (7.0 * cart[...,2] * cart[...,2] - 1.0)


def HardcodedSH4p3(cart) :
    # -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
    return -1.770131 * cart[...,0] * cart[...,2] * (cart[...,0] * cart[...,0] - 3.0 * cart[...,1] * cart[...,1])


def HardcodedSH4p4(cart) :
    # 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
    x2 = cart[...,0] * cart[...,0]
    y2 = cart[...,1] * cart[...,1]
    return 0.625836 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2))


def compute_RSH_coeffs(data, coords, order, sample_weight=None):
    '''
    data: [..., n_features] where [...] are data dimensions
    coords: [..., 2] or [..., 3]
    sample_weight: [..., n_features], defult all ones, will be normalised so that each feature channel sums to 4pi
    '''
    assert data.shape[:-1] == coords.shape[:-1]
    nfeatures = data.shape[-1]

    if sample_weight is None:
        sample_weight = torch.ones_like(data)

    data = data.reshape((-1, nfeatures))
    coords = coords.reshape((-1, coords.shape[-1]))
    sample_weight = sample_weight.reshape((-1, sample_weight.shape[-1]))

    # normalize
    data = data * sample_weight / sample_weight.sum(0) * 4 * pi

    coefficients = []
    for l in range(order+1):
        for m in range(-l, l+1):
            basis = rsh(l, m, coords)[..., None] #[..., 1]
            coefficients.append((basis*data).sum(0))
    return torch.stack(coefficients, dim=0) 

def reconstruct_from_RSH_coeffes(coords, coeffs):
    order = int(sqrt(coeffs.shape[0])) - 1
    assert (order+1)**2 == coeffs.shape[0]
    nfeatures = coeffs.shape[1]
    ndim = coords.shape[:-1]
    data = 0
    i = 0
    for l in range(order+1):
        for m in range(-l, l+1):
            basis = rsh(l, m, coords)[..., None] #[..., 1]
            data = data + basis * coeffs[i]
            i = i + 1
    return data.reshape(ndim + (nfeatures,))



if __name__ == '__main__':
    import cv2
    device = torch.device("cuda:0")
    img = torch.tensor(cv2.imread('pano4s.png')).to(torch.float32).to(device)

    height, width = img.shape[:2]

    azimuths = torch.arange(0, width, dtype=torch.float32, device=device) / width * np.pi * 2 - (width-1) / width * np.pi
    elevations = -torch.arange(0, height, dtype=torch.float32, device=device) / height * np.pi + (height-1) / height * np.pi / 2

    elevations, azimuths = torch.meshgrid(elevations, azimuths)
    dirs = torch.stack((
        torch.cos(elevations)*torch.sin(azimuths), #x
        torch.cos(elevations)*torch.cos(azimuths), #y
        torch.sin(elevations), #z
    ), dim=-1)

    weights = torch.cos(elevations[...,None])
    rsh_coeffs = compute_RSH_coeffs(img, dirs, 7, weights)
    img_recon = reconstruct_from_RSH_coeffes(dirs, rsh_coeffs)

    cv2.imwrite('pano4s_recon_rsh7.png', img_recon.clamp(0,255).cpu().numpy().astype(np.uint8))