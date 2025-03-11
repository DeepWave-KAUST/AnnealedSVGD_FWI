import gc
import torch
import numpy as np
import deepwave 
import math

def compute_gradient(
    model,
    x_s,
    x_r,
    nz,
    nx,
    dx,
    dt,
    freq,
    data_true,
    source_wavelet,
    log_likelihood,
    batch_size,
    data_normalization=None,
    device="cpu",
):
    """
    Compute the gradient of the loss function

    Args:
        model: velocity model
        x_s: source coordinates
        x_r: receiver coordinates
        nz: number of grid points in z direction
        nx: number of grid points in x direction
        dx: grid spacing
        dt: sample interval
        data_true: true data
        source_wavelet: source wavelet
        log_likelihood: log likelihood function
        data_normalization: data normalization
        device: device to use
    Returns:
        gradient of the loss function
    """
    model = (
        model.reshape(nz, nx) if len(model.shape) == 1 else model
    )
    # model.requires_grad = True

    running_loss = 0.0
    grad_loss = torch.zeros_like(model).to(device)
    batch_src_wvl = source_wavelet.to(device)
    batch_data_true = data_true.detach().to(device)
    batch_x_s = x_s.to(device)
    batch_x_r = x_r.to(device)
    data_pred = deepwave.scalar(model.to(device), dx, dt, source_amplitudes=batch_src_wvl, source_locations=batch_x_s, receiver_locations=batch_x_r, pml_freq=freq)[-1].to(device)
    
    if data_normalization is not None:
            batch_data_true = data_normalization(batch_data_true)
            data_pred = data_normalization(data_pred)
    
    loss = 1e3*log_likelihood(data_pred, batch_data_true)
    running_loss += loss.item()
    grad_loss += torch.autograd.grad(loss, model)[-1]

    gc.collect()
    torch.cuda.empty_cache()

    return running_loss, grad_loss.detach().cpu().numpy()


def compute_gradient_per_batch(model, grad_func):
    log_p = 0.0
    fwi_grad = np.zeros_like(model.detach().cpu().numpy())
    for i, m in enumerate(model):
        m.requires_grad_(True)
        loss, grad_m = grad_func(m)
        fwi_grad[i] = grad_m.ravel()
        log_p += loss
    log_p /= len(model)
    return log_p, fwi_grad


def compute_max_gradient(grad):
    return np.max(np.abs(grad))


def compute_max_gradient_per_batch(grad):
    assert len(grad.shape) == 2
    return np.max(np.abs(grad), axis=1, keepdims=True)