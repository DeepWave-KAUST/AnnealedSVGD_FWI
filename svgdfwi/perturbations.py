import torch
import numpy as np
import gstools as gs
from concurrent.futures import ThreadPoolExecutor
import os

def generate_random_field_perturbations(cfg, distribution='random', min_var=1, max_var=100, seed=20170590, len_scale=5, nu=2, device='cpu'):
    """
    Generates random field perturbations for a set of particles based on a variance of an specified distribution.
    
    Parameters:
        cfg (.yaml): Configuration containing simulation parameters.
    
        distribution (str): Type of distribution to sample variances ('random' or 'uniform').
        min_var (float): Minimum variance for perturbation sampling.
        max_var (float): Maximum variance for perturbation sampling.
        seed (int): Seed for random number generator.
        len_scale (float): Length scale for the Matern function.
        nu (float): Smoothness parameter for the Matern function.
        device (str): Device to perform computations on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: perturbations to be added, on the specified device.
    """
    num_particles = cfg.params.num_particles
    nz, nx = cfg.params.nz, cfg.params.nx
    zax = torch.arange(0, nz)
    xax = torch.arange(0, nx)

    # Sample variances based on the user-defined distribution
    if distribution == 'random':
        variances = np.random.rand(num_particles) * (max_var - min_var) + min_var
    elif distribution == 'uniform':
        variances=np.random.uniform(min_var,max_var,num_particles)
    else:
        raise ValueError("Unsupported distribution type. Use 'random' or 'uniform'.")

    variances = np.random.permutation(variances)  # Shuffle the variances

    # Setup random fields with gstools
    grf_seed = gs.random.MasterRNG(seed)
    grf = torch.zeros(num_particles, nz * nx, device=device)

    for i in range(num_particles):
        rf = gs.Matern(dim=2, var=variances[i], len_scale=len_scale, nu=nu)
        srf = gs.SRF(rf, seed=grf_seed())
        srf.set_pos([zax.numpy(), xax.numpy()], "structured")
        grf[i, :] = torch.from_numpy(srf().reshape(1, nz * nx)).float().to(device)

    return grf, variances




def generate_single_field(i, var, nz, nx, len_scale, nu, seed):
    """
    Generate a single random field using the Matérn covariance function.
    Detailed function documentation omitted for brevity.
    """
    zax = np.arange(0, nz)
    xax = np.arange(0, nx)
    rf = gs.Matern(dim=2, var=var, len_scale=len_scale, nu=nu)
    srf = gs.SRF(rf, seed=seed)
    srf.set_pos([zax, xax], "structured")
    field = srf()
    return i, field.reshape(1, nz * nx)
