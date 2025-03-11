import torch
import numpy as np
import math
import deepwave


def data_max_normalization(data):
    data_max, _ = data.abs().max(dim=0, keepdim=True)
    return data / (data_max + 1e-10)

def data_energy_normalization(data):
    data=data/torch.sqrt(data.sum(dim=0, keepdim=True)**2)
    return data

def add_noise(data, noise_level=0.1):
    """
    Adding Gaussian noise to the data
    Args:
        data: seismic data
        noise_level: noise level
    Returns:
        Noisy data
    """
    noise = noise_level * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data

def snr(x, x_est):
    """
    Compute the SNR in dB
    # Replace this snr by the pylops snr
    Args:
        x: original signal
        x_est: estimated signal
    Returns:
        SNR in dB
    """
    return 10.0 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))


def setup_fwi_acquisition(cfg, device='cpu'):
    # Extract parameters from configuration
    nz = cfg.params.nz
    nx = cfg.params.nx
    dx = cfg.params.dx
    
    xax = torch.arange(0, cfg.params.nx)
    zax = torch.arange(0, cfg.params.nz)

    n_shots = cfg.params.ns
    n_sources_per_shot = cfg.params.num_sources_per_shot
    first_source = cfg.params.first_source
    source_depth = cfg.params.source_depth

    n_receivers_per_shot = cfg.params.num_receivers_per_shot
    first_receiver = cfg.params.first_receiver
    receiver_depth = cfg.params.receiver_depth

    freq = cfg.params.peak_freq
    nt = cfg.params.nt
    dt = cfg.params.dt
    peak_time = cfg.params.peak_time / freq
    tax = torch.arange(0, nt) * dt

    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long, device=device)
    source_locations[..., 0] = source_depth
    source_locations[:, 0, 1] = torch.linspace(0, nx - 2, n_shots) + first_source

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.long, device=device)
    receiver_locations[..., 0] = receiver_depth
    receiver_locations[:, :, 1] = (
        torch.linspace(0, nx - 2, n_receivers_per_shot) +
        first_receiver
    ).repeat(n_shots, 1)

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1).to(device)
    )

    return source_locations, receiver_locations, source_amplitudes, tax, xax, zax

def setup_msfwi_acquisition(cfg, step, device='cpu'):
    # Extract parameters from configuration
    nz = cfg.params.nz
    nx = cfg.params.nx
    dx = cfg.params.dx
    
    xax = torch.arange(0, cfg.params.nx)
    zax = torch.arange(0, cfg.params.nz)

    n_shots = cfg.params.ns
    n_sources_per_shot = cfg.params.num_sources_per_shot
    first_source = cfg.params.first_source
    source_depth = cfg.params.source_depth

    n_receivers_per_shot = cfg.params.num_receivers_per_shot
    first_receiver = cfg.params.first_receiver
    receiver_depth = cfg.params.receiver_depth

    freq = cfg.params.peak_freq[step]
    nt = cfg.params.nt
    dt = cfg.params.dt
    peak_time = cfg.params.peak_time / freq
    tax = torch.arange(0, nt) * dt

    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long, device=device)
    source_locations[..., 0] = source_depth
    source_locations[:, 0, 1] = torch.linspace(0, nx - 2, n_shots) + first_source

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.long, device=device)
    receiver_locations[..., 0] = receiver_depth
    receiver_locations[:, :, 1] = (
        torch.linspace(0, nx - 2, n_receivers_per_shot) +
        first_receiver
    ).repeat(n_shots, 1)

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1).to(device)
    )

    return source_locations, receiver_locations, source_amplitudes, tax, xax, zax


def save_dict_as_compressed_npz(file_name, data_dict):
    """
    Save a dictionary with array values to a compressed .npz file, ensuring all data is in float32 format.
    Converts PyTorch tensors to NumPy arrays and casts to float32 automatically.

    Parameters:
     file_name: str, the name of the file to save the data to.
     data_dict: dict, a dictionary with values as arrays (NumPy or PyTorch tensors).
    """
    # Prepare data for saving: convert PyTorch tensors to NumPy arrays and cast to float32 if necessary
    save_dict = {}
    for key, value in data_dict.items():
        # Handle PyTorch tensors
        if isinstance(value, torch.Tensor):
            converted_value = value.cpu().numpy().astype(np.float32)
        # Handle NumPy arrays and cast to float32 if not already
        elif isinstance(value, np.ndarray) and value.dtype != np.float32:
            converted_value = value.astype(np.float32)
        else:
            converted_value = value  # For non-numeric types, no conversion applied

        save_dict[key] = converted_value

    # Save as a compressed .npz file, using the dictionary keys as variable names
    np.savez_compressed(file_name, **save_dict)


def calculate_bins(data):
    N = len(data)
    sqrt_bins = int(np.sqrt(N))
    sturges_bins = int(np.log2(N)) + 1
    rice_bins = int(2 * N ** (1/3))

    return {
        "Square_Root": sqrt_bins,
        "Sturges": sturges_bins,
        "Rice": rice_bins
    }
