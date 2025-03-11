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
from tqdm import tqdm

import imageio
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hydra import initialize, compose
from scipy.ndimage import gaussian_filter1d



# set seed
np.random.seed(10)
random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_files = 35+1
print('Starting plots...')

# for i in tqdm(range(num_files), desc="Processing files"):
exp_id = f"{7:02d}"
with initialize(version_base=None, config_path="../config/multiscale"):
    cfg = compose(f"{exp_id}_multiscale")

print(f"os.getcwd(): {os.getcwd()}")

## Load true model
model_true = np.load('../data/marmousi_portion.npy', allow_pickle=True).astype('float32')

## Load results file
results = np.load(f'../results/multiscale/{exp_id}_multiscale/{exp_id}_results_svgd.npz', allow_pickle=True)
num_epochs, num_particles, particle_dimensions = results['updates'].shape
step = 0
source_locations, receiver_locations, source_amplitudes, time_axis, xax, zax = setup_msfwi_acquisition(cfg, step, device=device)  
nz = cfg.params.nz
nx = cfg.params.nx
dx = cfg.params.dx
m_vmin = cfg.params.m_vmin
m_vmax = cfg.params.m_vmax
std_min = 0.0
std_max = 400.0

## plots config
fps = 40

out_path = f'../videos/multiscale/{exp_id}_multiscale/'
temp_path = f'../temp_frames_mean/multiscale/{exp_id}_multiscale/'
mean_video_path = f'{exp_id}_mean_updates.mp4'
std_video_path = f'{exp_id}_std_updates.mp4'
kernel_video_path = f'{exp_id}_kernel_updates.mp4'
pca_video_path = f'{exp_id}_pca_updates.mp4'

os.makedirs(out_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# ## Videos Global
# # Video for mean update
# make_video_attribute(results['updates_mean'].astype('float32'), vmin=m_vmin, vmax=m_vmax, cmap='terrain', title='Mean', 
#                         colorbar=True, fig_size=(8, 4), epochs=num_epochs, video_dir=out_path, 
#                         nz=nz, nx=nx,
#                         temp_dir=temp_path, video_name=mean_video_path, fps=fps)

# # Video for std update
# make_video_attribute(results['updates_std'].astype('float32'), vmin=std_min, vmax=std_max, cmap='Reds', title='Std', 
#                         colorbar=True, fig_size=(8, 4), epochs=num_epochs, video_dir=out_path, 
#                         nz=nz, nx=nx,
#                         temp_dir=temp_path, video_name=std_video_path, fps=fps)

# # Video for kernel update
# make_video_attribute(results['kernels'].astype('float32'), vmin=0, vmax=1, cmap='Blues', title='Kernel', 
#                         colorbar=True, fig_size=(4, 4), epochs=num_epochs, video_dir=out_path, 
#                         nz=num_particles, nx=num_particles,
#                         temp_dir=temp_path, video_name=kernel_video_path, fps=fps)


# ## Plot 10 initial particles
# print('Initial particles...')
# fig, axs = plt.subplots(2, 5, figsize=(14, 4)) # 1 row, 10 columns

# for i, ax in enumerate(axs.flatten()):
#     data = results['updates'][0][i].reshape(nz, nx)
#     im = ax.imshow(data, aspect='auto', vmin=m_vmin, vmax=m_vmax, cmap='terrain')
#     ax.set_title(f'Particle {i+1}')
#     ax.axis('off')

# # Add a general colorbar
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5]) # Add_axes([left, bottom, width, height])
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the layout to make room for the colorbar
# plt.savefig(f'{out_path}{exp_id}_initial_particles.png', dpi=150)
# # plt.show()


# ## Plot 10 last particles
# print('Last particles...')
# fig, axs = plt.subplots(2, 5, figsize=(14, 4)) # 1 row, 10 columns

# for i, ax in enumerate(axs.flatten()):
#     data = results['updates'][-1][i].reshape(nz, nx)
#     im = ax.imshow(data, aspect='auto', vmin=m_vmin, vmax=m_vmax, cmap='terrain')
#     ax.set_title(f'Particle {i+1}')
#     ax.axis('off')

# # Add a general colorbar
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5]) # Add_axes([left, bottom, width, height])
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the layout to make room for the colorbar
# plt.savefig(f'{out_path}{exp_id}_last_particles.png', dpi=150)
# # plt.show()


# ## Plot relative error 10 particles
# #Compute the relative error
# print('Relative error...')
# relative_error = np.abs(results['updates'][-1].astype('float32')-model_true.reshape(1,-1))

# ## Relative error |mparticle-mtrue-|/std
# fig, axs = plt.subplots(2, 5, figsize=(14, 4)) # 1 row, 10 columns

# for i, ax in enumerate(axs.flatten()):
#     data = relative_error[i].reshape(nz, nx)/results['updates_std'][-1].reshape(nz, nx).astype('float32')
#     im = ax.imshow(data, aspect='auto', cmap='viridis') #, vmin=0, vmax=100)
#     ax.set_title(f'Particle {i+1}')
#     ax.axis('off')

# # Add a general colorbar
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5]) # Add_axes([left, bottom, width, height])
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust the layout to make room for the colorbar
# plt.savefig(f'{out_path}{exp_id}_relative_error_particles.png', dpi=150)
# # plt.show()

# ## Plotting Loss
# print('Loss...')
# plt.figure(figsize=(6, 5))
# plt.semilogy(results['epoch_loss'].astype('float32'))
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.tight_layout()
# plt.savefig(f'{out_path}{exp_id}_loss.png', dpi=150)


# ## Plotting bandwidth
# print('Bandwidth...')
# plt.figure(figsize=(6, 5))
# plt.plot(results['sigmas'].astype('float32'))
# plt.xlabel('epochs')
# plt.ylabel('Bandwidth')
# plt.tight_layout()
# plt.savefig(f'{out_path}{exp_id}_bandwidth.png', dpi=150)


## Marginal plots
print('Marginals...')
idxs = [30, nx // 2, nx - 30]
# depth = np.arange(nz)
depth = np.linspace(0, (nz-1)*dx*1e-3, nz)

# Apply Gaussian smoothing to the standard deviations
sigma = 2 # You can adjust this parameter to control the smoothing level
smoothed_std = gaussian_filter1d(results['updates_std'][-1].reshape(nz, nx).astype('float32'), sigma, axis=0)

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(7, 8))
for i, ax in enumerate(axs):
    mean = results['updates_mean'][-1].reshape(nz, nx)[:, idxs[i]]
    std = smoothed_std[:, idxs[i]]
    
    ax.plot(mean, depth, label='Mean', color='blue')
    ax.fill_betweenx(depth, mean - std, mean + std, color='royalblue', alpha=0.4, label='68.3 %')
    ax.fill_betweenx(depth, mean - 2*std, mean + 2*std, color='royalblue', alpha=0.2, label='95.5 %')
    ax.fill_betweenx(depth, mean - 3*std, mean + 3*std, color='gray', alpha=0.1, label='99.7 %')
    ax.plot(model_true.reshape(nz, nx)[:, idxs[i]], depth, label='True Model', color='black', linestyle='dashed')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    
    ax.set_xlabel(r'Velocity $[m/s]$')
    if i == 0:
        ax.set_ylabel('Depth [km]')
    ax.set_title(f'Pseudo-well x={idxs[i]*dx*1e-3} km')
    ax.set_aspect('auto', 'datalim')
    ax.invert_yaxis()

ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{out_path}{exp_id}_marginal_pseudo_wells_multiscale.png', dpi=150)





# #pixel location for marginals
# num_bins = 10
# loc_0 = (30, 30)
# loc_1 = (nz//2, nx//2)
# loc_2 = (65, 160)


# fig, axs = plt.subplots(1, 3, sharey=False, figsize=(15, 5))
# axs[0].hist(results['updates'][0].reshape(num_particles, nz, nx)[:, loc_0[0], loc_0[1]],
#             bins=num_bins, density=True, alpha=0.5, label='Initial')
# axs[0].hist(results['updates'][-1].reshape(num_particles, nz, nx)[:, loc_0[0], loc_0[1]],
#             bins=num_bins, density=True, alpha=0.5, label='SVGD')
# axs[0].axvline(model_true.reshape(nz,nx)[loc_0[0], loc_0[1]], c='k', lw=2, label='ground truth')
# axs[0].set_xlabel(r'Velocity $[m/s]$')
# axs[0].set_ylabel('PDF')
# axs[0].set_title(f'(z={loc_0[1]*dx*1e-3}, x={loc_0[0]*dx*1e-3}) km')
# axs[0].set_aspect('auto', 'datalim')
# axs[0].legend(loc='upper right')
# axs[0].set_xlim([1500, 2000])

# axs[1].hist(results['updates'][0].reshape(num_particles, nz, nx)[:, loc_1[0], loc_1[1]],
#             bins=num_bins, density=True, alpha=0.5, label='Initial')
# axs[1].hist(results['updates'][-1].reshape(num_particles, nz, nx)[:, loc_1[0], loc_1[1]],
#             bins=num_bins, density=True, alpha=0.5, label='SVGD')
# axs[1].axvline(model_true.reshape(nz,nx)[loc_1[0], loc_1[1]], c='k', lw=2, label='ground truth')
# axs[1].set_xlabel(r'Velocity $[m/s]$')
# axs[1].set_ylabel('PDF')
# axs[1].set_title(f'(z={loc_1[0]*dx*1e-3}, x={loc_1[1]*dx*1e-3}) km')
# axs[1].set_aspect('auto', 'datalim')
# axs[1].legend(loc='upper right')
# axs[1].set_xlim([1800, 2300])

# axs[2].hist(results['updates'][0].reshape(num_particles, nz, nx)[:, loc_2[0], loc_2[1]],
#             bins=num_bins, density=True, alpha=0.5, label='Initial')
# axs[2].hist(results['updates'][-1].reshape(num_particles, nz, nx)[:, loc_2[0], loc_2[1]],
#             bins=num_bins, density=True, alpha=0.5, label='SVGD')
# axs[2].axvline(model_true.reshape(nz,nx)[loc_2[0], loc_2[1]], c='k', lw=2, label='ground truth')
# axs[2].set_xlabel(r'Velocity $[m/s]$')
# axs[2].set_ylabel('PDF')
# axs[2].set_title(f'(z={loc_2[0]*dx*1e-3}, x={loc_2[1]*dx*1e-3}) km')
# axs[2].set_aspect('auto', 'datalim')
# axs[2].legend(loc='upper right')
# axs[2].set_xlim([2000, 4000])

# plt.tight_layout()
# plt.savefig(f'{out_path}{exp_id}_marginal_pixels_multiscale.png', dpi=150)

# ## PCA - 2 components 
# print('PCA and clustering...')
# particles = np.copy(results['updates'].astype('float32'))

# # Reshape particles to combine all epochs into one 2D array (num_epochs * num_particles, particle_dimensions)
# particles_reshaped = particles.reshape(num_epochs * num_particles, particle_dimensions)

# # Initialize PCA
# components = 2
# pca = PCA(n_components=components)

# # Apply PCA to the combined particles data
# pca_result = pca.fit_transform(particles_reshaped)

# # Reshape the PCA result back to (num_epochs, num_particles, components)
# pca_data_2comp = pca_result.reshape(num_epochs, num_particles, components)

# # Make video
# make_video_pca(pca_data_2comp, title='PCA 2 components', fig_size=(4, 4), epochs=num_epochs, 
#             video_dir=out_path, temp_dir=temp_path, 
#             video_name=pca_video_path, fps=fps, marker_size=100)



# ## PCA n-1 components  (n=particles number)
# # Perform PCA on reshaped data
# pca = PCA(n_components=num_particles-1)
# pca.fit(particles_reshaped)

# # Get the components
# components = pca.components_

# # Calculate the explained variance per epoch
# explained_variance_per_epoch = np.zeros((num_epochs, components.shape[0]))

# for epoch in range(num_epochs):
#     epoch_data = particles[epoch]
#     transformed_data = pca.transform(epoch_data)
#     explained_variance = np.var(transformed_data, axis=0)
    
#     # total_variance = np.sum(np.var(epoch_data, axis=0))
#     # explained_variance_ratio = explained_variance / total_variance
    
#     explained_variance_per_epoch[epoch] = explained_variance #explained_variance_ratio
    

# plt.figure(figsize=(6, 5))
# for component in range(components.shape[0]):
#     plt.semilogy(range(num_epochs), explained_variance_per_epoch[:, component])
# plt.xlabel('Epochs')
# plt.ylabel('Variance')
# plt.title('Explained Variance per component')
# plt.tight_layout()
# plt.savefig(f'{out_path}{exp_id}_explained_variance.png', dpi=150)
# # plt.show()

# ## Clustering 
# # Number of clusters (assuming you know this a priori or have determined it)
# num_clusters = 2

# # Initialize KMeans
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# # Fit KMeans to the data
# kmeans.fit(particles[-1])

# # Get cluster assignments
# cluster_assignments = kmeans.labels_

# # Separate particles by cluster
# particles_by_cluster = {i: particles[-1][cluster_assignments == i] for i in range(num_clusters)}


# # Visualize the PCA data with cluster assignments for the last epoch
# plt.figure(figsize=(6, 5))
# for cluster in range(num_clusters):
#     cluster_points = pca_data_2comp[-1][cluster_assignments == cluster]
#     cluster_size = cluster_points.shape[0]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster} ({cluster_size} particles)', s=50, alpha=0.6)

# plt.title('PCA of Particles (Last Epoch) with Clustering')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{out_path}{exp_id}_pca_clustering.png', dpi=150)
# # plt.show()


# # Plot global/cluster mean and std and 
# print('Statistics by cluster...')
# #Global
# global_mean = particles[-1].astype('float32').mean(axis=0)
# global_std = particles[-1].astype('float32').std(axis=0)

# #Cluster_0
# mean_cluster_0 = particles_by_cluster[0].mean(0)
# std_cluster_0 = particles_by_cluster[0].std(0)

# #Cluster_1
# mean_cluster_1 = particles_by_cluster[1].mean(0)
# std_cluster_1 = particles_by_cluster[1].std(0)


# plot_path = f'{out_path}{exp_id}_mean_std_clusters.png'

# plot_multiple_means_stds(global_mean, global_std, mean_cluster_0, std_cluster_0, mean_cluster_1, std_cluster_1, 
#                             cfg, m_vmin, m_vmax, std_min, std_max, xax, zax, plot_path, save_plots=True)

print("Plots done habibi!")

# # run the main function
# if __name__ == "__main__":
#     main()

 