
import numpy as np
import torch
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RBF(torch.nn.Module):
    """
        Initializes the RBF_FWI class.

        Args:
            sigma (float, optional): Bandwidht of the RBF kernel.
    """
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)

        if self.sigma is None:
            h = torch.median(dists) / (2 * math.log(X.size(0) + 1))
            sigma = math.sqrt(h)
        
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        
        else:
            sigma = self.sigma

        k = torch.exp(-dists / sigma**2 / 2)
        der = (d * k[:, :, None]).sum(axis=0) / sigma**2
        
    
        return k, der, sigma 


class IMQ(torch.nn.Module):
    """Inverse Multiquadric Kernel (IMQ) with beta = -1/2 and c =1 Following
        Measuring sample Quality with Kernels by Gorham & Mackey, 2020
        """
    def __init__(self, sigma=None):
        super(IMQ, self).__init__()
        self.sigma = sigma

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)
       
        # Apply the median
        if self.sigma is None:
            sigma = torch.median(dists)
            
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        else:
            sigma = self.sigma

        k = 1. / torch.sqrt(1. + dists / sigma)
        dxkxy = .5 * k / (1. + dists / sigma)
        der = (d * dxkxy[:, :, None]).sum(axis=0) * 2. / sigma
        
        # Delete the intermediate variables to free memory
        del d, dists, dxkxy
        torch.cuda.empty_cache()  # Clear memory cache if necessary

        return k, der, torch.sqrt(sigma)
    
    
def find_partially_isolated_particles_symmetric(kernel_matrix, threshold=1e-5, percentage=90):
    """
    Identify particles where at least a specified percentage of off-diagonal elements
    are below a given threshold. Assumes the kernel_matrix is symmetric.
    Parameters:
        kernel_matrix (numpy.ndarray): A symmetric square matrix representing the interaction
                                       between particles.
        threshold (float): A threshold below which values are considered to be near zero.
        percentage (float): The minimum percentage of off-diagonal elements that must
                            be below the threshold for a particle to be considered.
    Returns:
        numpy.ndarray: An array of indices representing particles that meet the criterion.
    """
    num_particles = kernel_matrix.shape[0]
    threshold_percentage = percentage / 100.0
    isolated_indices = []
    
    for i in range(num_particles):
        # Considering only the row since matrix is symmetric
        elements = np.abs(kernel_matrix[i, :])
        elements[i] = np.inf  # Ignore diagonal by setting it to infinity
        # Calculate the percentage of elements below the threshold
        
        below_threshold = np.sum(elements < threshold) / (num_particles - 1)
        
        # Check if the row meets the percentage criterion
        if below_threshold >= threshold_percentage:
            isolated_indices.append(i)
            
    return np.array(isolated_indices)


def introduce_variations(original_particles, device='cpu', variation_scale=0.1):
    """
    Introduce slight variations to the particle data to avoid exact duplicates.
    Parameters:
        original_particles (torch.Tensor): Tensor of particle data that needs variation.
        device (str): The device tensors are on ('cpu' or 'cuda').
        variation_scale (float): The scale of random noise to introduce.
    Returns:
        torch.Tensor: Tensor containing the varied particle data.
    """
    noise = torch.randn_like(original_particles) * variation_scale
    return original_particles + noise.to(device)


def replace_particles(kernel_matrix, particle_data, threshold=1e-1, percentage=90, device='cpu', kind='isolated'):
    """
    Replace isolated particles with varied versions of interacting particles.
    Parameters:
        kernel_matrix (numpy.ndarray): The interaction matrix of the particles.
        particle_data (torch.Tensor): The tensor containing data of all particles.
        threshold (float): Threshold below which interactions are considered negligible.
        percentage (float): Percentage of negligible interactions to qualify as isolated.
        device (str): The device to perform computations on ('cpu' or 'cuda').
        kind (str): The kind of particles to replace ('isolated'--replace isolated particles or 
                    'interacting' -- replace the "good" particles).
    Returns:
        torch.Tensor: Updated particle data with isolated particles replaced.
    """
    # Identify isolated and interacting particles
    isolated_indices = find_partially_isolated_particles_symmetric(kernel_matrix, threshold, percentage)
    interacting_indices = [i for i in range(kernel_matrix.shape[0]) if i not in isolated_indices]
    
    # Ensure there are interacting particles to copy from
    if not interacting_indices:
        raise ValueError("No interacting particles available for replacement.")
    
    cp_particle_data = np.copy(particle_data)
    
    if kind=='isolated':
        # Randomly select replacements from the interacting particles and apply variations
        
        for isolated_idx in isolated_indices:
            # Randomly choose an index from the interacting particles
            chosen_idx = np.random.choice(interacting_indices)
            
            # Get a particle and introduce variations
            varied_particle = introduce_variations(particle_data[chosen_idx], device=device, variation_scale=0.100)
            
            # Replace the isolated particle
            cp_particle_data[isolated_idx] = varied_particle
            
    elif kind=='interacting':
        #Here we duplicate the number of isolated indices to have more particles to replace
        for interacting_idx in np.random.permutation(interacting_indices)[:isolated_indices.shape[0]*2]:
            
            # Randomly choose an index from the interacting particles
            chosen_idx = np.random.choice(isolated_indices)

            # Get a particle and introduce variations
            varied_particle = introduce_variations(particle_data[chosen_idx], device=device, variation_scale=0.100)
            
            # Replace the isolated particle
            cp_particle_data[interacting_idx] = varied_particle
    else:
        raise ValueError("Unsupported kind option. Use 'isolated' or 'interacting'.")
    
    return cp_particle_data
    
    
