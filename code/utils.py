import random
import numpy as np
import torch
import skfmm


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False


def set_device():
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        print("No GPU available!")
    return device


def eikonal_grid(ox, dx, nx, oz, dz, nz):
    """Create regular grid
    """
    x = np.arange(nx) * dx + ox
    z = np.arange(nz) * dz + oz

    X, Z= np.meshgrid(x, z)

    return x, z, X, Z


def eikonal_constant(ox, dx, nx, oz, dz, nz, xs, zs, v):
    """Eikonal solution in constant velocity

    Compute analytical eikonal solution and its spatial derivatives
    for constant velocity model

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    # Analytical solution
    dana = np.sqrt((X - xs) ** 2 + (Z - zs) ** 2)
    tana = dana / v

    # Derivatives of analytical solution
    tana_dx = (X - xs) / (dana * v)
    tana_dz = (Z - zs) / (dana * v)

    return tana, tana_dx, tana_dz


def eikonal_gradient(ox, dx, nx, oz, dz, nz, xs, zs, v0, k_vertical, k_horizontal):
    """Eikonal solution in gradient velocity

    Compute analytical eikonal solution for gradient velocity model

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    # Velocity
    v = v0 + k_vertical * (Z - oz) + k_horizontal * (X - ox)
    vs = v[(X==xs)&(Z==zs)]
    
    # Analytical solution
    dist2 = (X - xs) ** 2 + (Z - zs) ** 2
    tana = (1. / np.sqrt(k_vertical**2 + k_horizontal**2)) * np.arccosh(1 + ((k_vertical**2 + k_horizontal**2) * dist2) / (2 * v * vs))

    return tana


def eikonal_fmm(ox, dx, nx, oz, dz, nz, xs, zs, v):
    """Fast-marching method eikonal solution

    Compute eikonal solution using the fast-marching method for benchmark

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    phi = np.ones((nz, nx))
    phi[int(zs // dz), int(xs // dx)] = -1.
    teik = skfmm.travel_time(phi, v, dx=(dz, dx))

    return teik


def remove_source(X, Z, xs, zs, v, tana, tana_dx, tana_dz):
    """Remove source from grids

    Remove element corresponding to the source index due to the fact that
    the analytical derivatives for the traveltime are undefined

    """
    # Find source index
    isource = (X.ravel() == xs) & (Z.ravel() == zs)

    X_nosrc, Z_nosrc = X.ravel()[~isource], Z.ravel()[~isource]
    v_nosrc, tana_nosrc = v.ravel()[~isource], tana.ravel()[~isource]
    tana_dx_nosrc, tana_dz_nosrc = tana_dx.ravel()[~isource], tana_dz.ravel()[~isource]

    return X_nosrc, Z_nosrc, v_nosrc, tana_nosrc, tana_dx_nosrc, tana_dz_nosrc