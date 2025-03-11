# Experiments for FWI SVGD
import warnings
warnings.filterwarnings('ignore')
import os
import hydra
import time
import random
import torch
import gc
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from tqdm import tqdm
import gstools as gs
from gstools.random import MasterRNG

from svgdfwi.fwi import *
from svgdfwi.kernel import *
from svgdfwi.utils import *
from svgdfwi.annealing import *
from svgdfwi.svgd import *
from svgdfwi.plots import *
from svgdfwi.perturbations import *




# set seed
np.random.seed(10)
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup config and main function
@hydra.main(config_path="../../config/exp", config_name="21_exp")
def main(cfg):
    print(f"os.getcwd(): {os.getcwd()}")
    
    # FWI set up
    model_true = torch.from_numpy(np.load(f'{cfg.paths.path}{cfg.files.velocity_model}', allow_pickle=True).astype('float32')).to(device)
    m_vmin, m_vmax = (cfg.params.m_vmin, cfg.params.m_vmax)
    
    #Grid
    source_locations, receiver_locations, source_amplitudes, time_axis, xax, zax = setup_fwi_acquisition(cfg, device=device)
    nz = cfg.params.nz
    nx = cfg.params.nx
    dx = cfg.params.dx

    freq = cfg.params.peak_freq
    nt = cfg.params.nt
    dt = cfg.params.dt

    data_obs = deepwave.scalar(
    model_true,
    dx, dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_freq=freq,
    )[-1]

    # Generate noisy data
    observed_data = add_noise(
        data_obs,
        noise_level=0.5,
    )

    # Compute data SNR
    data_noisy_snr = snr(
        data_obs.detach().cpu().numpy(), observed_data.detach().cpu().numpy()
    )

    print(f"Noisy data SNR: {data_noisy_snr:.2f} dB")

    # Initial model
    initial_model = torch.tensor(1/gaussian_filter(1/model_true.detach().cpu().numpy(),30)).to(device)
    
    # Create perturbations
    print(f"Creating perturbations using GRF...")
    grf, variances = generate_random_field_perturbations(cfg, distribution='uniform', 
                                                        min_var=10, max_var=2000, 
                                                        seed=20170590, len_scale=5, 
                                                        nu=2, device=device)


    # Create set of initial particles
    X_init = initial_model.clone()
    X_init = X_init.reshape(1, -1).repeat(cfg.params.num_particles, 1).to(device)
    X_init = X_init + grf
    X_init.requires_grad = True
        
    
    # Optimization parameters
    loss_fn = torch.nn.MSELoss()
    batch_size=cfg.params.batch_size
    data_normalization=None

    grad_func = lambda x: compute_gradient(
        x,
        source_locations,
        receiver_locations,
        nz,
        nx,
        dx,
        dt,
        freq,
        observed_data,
        source_amplitudes,
        loss_fn,
        batch_size,
        data_normalization,
        device,
    )
    
    optimizer = torch.optim.Adam([X_init],lr=cfg.params.learning_rate)
    loss_fn = torch.nn.MSELoss()
    scheduler = None
    n_iterations=cfg.params.num_iterations
    
    # Temperature annealing
    # Annealed tanh SVGD with cutoff
    alpha = alpha_tanh_cutoff(cfg.params.num_iterations, p=2, cutoff=0.8, device=device)
    
    
    # Initialize kernel class
    K = RBF()
    

    ## Create directories if they dont exist 
    os.makedirs(f"{cfg.paths.out_path}", exist_ok=True)
    os.makedirs(f"{cfg.paths.figure_path}", exist_ok=True)
    
    # set up figure
    fig_path = f"{cfg.paths.figure_path}"
    

    # start SVGD
    print(f"Starting inversion with {cfg.params.num_particles} particles")
    t_start = time.time()
    
    # Initialize the  kernel and SVGD_FWI class
    svgd = SVGD(X_init, K, alpha, optimizer, scheduler, device=device)

    # Dictionary to save all variables
    results_svgd = {
        "epoch_loss": [],
        "updates": [],
        "updates_mean": [],
        "updates_std": [],
        "kernels": [],
        "sigmas": [],
    }

    # epochs = tqdm(range(cfg.params.num_iterations))
    epochs = tqdm(range(cfg.params.num_iterations))

    for iteration in epochs:
        
        # Convert X_init to CPU numpy array once and reuse
        X_init_np = X_init.detach().clone().cpu().numpy()
        

        # FWI model update and gradient computation
        running_loss=0
        running_loss, grad = compute_gradient_per_batch(X_init, grad_func)
        
        grad_collection = torch.zeros_like(X_init)
        for ig, sgrad in enumerate(grad):
            smooth_grad = sgrad.reshape(nz, nx)
            smooth_grad = torch.from_numpy(smooth_grad).to(device)
            grad_collection[ig] = smooth_grad.ravel()

        if iteration == 0:
            gmax = compute_max_gradient_per_batch(grad)
            gmax = torch.tensor(gmax).to(device=device)


        # svgd.step(X_init, grad_collection, m_vmin, m_vmax, iteration, gmax, EMA=None)
        
            
        # --------------------------------------
        # Using exponential moving average to control the sigma 
        if iteration == 0:
            svgd.step(X_init, grad_collection, m_vmin, m_vmax, iteration, gmax, EMA=None)

        # Adding Exponential moving average to the sigma
        EMA = [svgd.sigma, 0.7]
        svgd.step(X_init, grad_collection, m_vmin, m_vmax, iteration, gmax, EMA)

        
        ## --------------------------------------
        ## Kernel assisted particle replacement
        # if iteration % 10 == 0:
        #     replaced_particles = replace_particles(kernel_matrix=svgd.K_XX.detach().cpu().numpy(), 
        #                               particle_data=X_init.detach().clone().cpu(), 
        #                               threshold=1e-1, percentage=80, 
        #                               device='cpu', kind='isolated')
        #     with torch.no_grad():
        #         X_init.data = torch.from_numpy(replaced_particles).to(device)

        
        # Collect results
        results_svgd["updates"].append(X_init_np.astype(np.float16))
        results_svgd["updates_mean"].append(X_init_np.mean(0).astype(np.float16))
        results_svgd["updates_std"].append(X_init_np.std(0).astype(np.float16))
        results_svgd["sigmas"].append(np.float16(svgd.sigma)) 
        results_svgd["epoch_loss"].append(np.float16(running_loss))
        results_svgd["kernels"].append(svgd.K_XX.detach().cpu().numpy().astype(np.float16))



        if torch.isnan(X_init).any():
            print("NaN detected in X_init")

        # Perform garbage collection less frequently
        if iteration % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

            plot_update_models(results_svgd["updates_mean"], results_svgd["updates_std"], 
                               results_svgd["kernels"], cfg, m_vmin, m_vmax, xax, zax, 
                               save_plots=True, 
                               mean_plot_filename=f"{fig_path}mean{iteration}.png", 
                               std_plot_filename=f"{fig_path}std_{iteration}.png",
                               kernel_plot_filename=f"{fig_path}kernel_{iteration}.png")


        # Update tqdm postfix
        epochs.set_postfix(iter=iteration, loss=running_loss)


    t_end = time.time()
    print("Runtime:", (t_end - t_start) / 60, "minutes")

    
    #Save the dictionary 
    save_dict_as_compressed_npz(f"{cfg.paths.out_path}21_results_svgd.npz", results_svgd)
    print(f"Saved results to {cfg.paths.out_path}")


# run the main function
if __name__ == "__main__":
    main()

 