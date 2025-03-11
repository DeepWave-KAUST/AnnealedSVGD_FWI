import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import shutil
from tqdm import tqdm
import seaborn as sns

def plot_mean_std_models(updates_mean, updates_std, cfg, m_vmin, m_vmax, x, z):
    """
    Plots the current mean and standard deviation models.

    Parameters:
    - updates_mean: A list/array containing the mean updates for the models.
    - updates_std: A list/array containing the standard deviation updates for the models.
    - cfg: A configuration object with attributes `params.nz` and `params.nx` for model dimensions.
    - m_vmin, m_vmax: The minimum and maximum values to use for the color scale of the mean model plot.
    - x, z: Coordinate arrays for the horizontal and vertical axes, respectively.
    """

    # Plot for the mean model
    plt.figure(figsize=(8, 4))
    plt.imshow(
        updates_mean[-1].reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("Current Mean Model")
    plt.colorbar(shrink=0.5)
    plt.show()

    # Plot for the standard deviation model
    plt.figure(figsize=(8, 4))
    gr = updates_std[-1].reshape(cfg.params.nz, cfg.params.nx)
    g_min, g_max = np.percentile(gr, [2, 98])
    plt.imshow(
        gr, 
        cmap="Reds", 
        vmin=g_min, 
        vmax=g_max,
       extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("Current Std Model")
    plt.colorbar(shrink=0.5)
    plt.show()

def plot_mean_std_fast(updates_mean, updates_std, cfg, m_vmin, m_vmax, x, z):
    """
    Plots the current mean and standard deviation models.

    Parameters:
    - updates_mean: A list/array containing the mean updates for the models.
    - updates_std: A list/array containing the standard deviation updates for the models.
    - cfg: A configuration object with attributes `params.nz` and `params.nx` for model dimensions.
    - m_vmin, m_vmax: The minimum and maximum values to use for the color scale of the mean model plot.
    - x, z: Coordinate arrays for the horizontal and vertical axes, respectively.
    """

    # Plot for the mean model
    plt.figure(figsize=(8, 4))
    plt.imshow(
        updates_mean.reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("Current Mean Model")
    plt.colorbar(shrink=0.5)
    plt.show()

    # Plot for the standard deviation model
    plt.figure(figsize=(8, 4))
    gr = updates_std.reshape(cfg.params.nz, cfg.params.nx)
    g_min, g_max = np.percentile(gr, [2, 98])
    plt.imshow(
        gr, 
        cmap="Reds", 
        vmin=g_min, 
        vmax=g_max,
       extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("Current Std Model")
    plt.colorbar(shrink=0.5)
    plt.show()


def plot_update_models(updates_mean, updates_std, kernel, cfg, m_vmin, m_vmax, x, z, 
                       save_plots=False, mean_plot_filename='mean_model.png', 
                       std_plot_filename='std_model.png',
                       kernel_plot_filename='kernel_model.png'):
    """
    Plots and optionally saves the current mean and standard deviation models.

    Parameters:
    - updates_mean: A list/array containing the mean updates for the models.
    - updates_std: A list/array containing the standard deviation updates for the models.
    - cfg: A configuration object with attributes `params.nz` and `params.nx` for model dimensions.
    - m_vmin, m_vmax: The minimum and maximum values to use for the color scale of the mean model plot.
    - x, z: Coordinate arrays for the horizontal and vertical axes, respectively.
    - save_plots: A boolean indicating whether to save the plots to files.
    - mean_plot_filename: Filename for saving the mean model plot.
    - std_plot_filename: Filename for saving the standard deviation model plot.
    """

    # Plot for the mean model
    plt.figure(figsize=(8, 4))
    plt.imshow(
        updates_mean[-1].reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("mean")
    plt.colorbar(shrink=0.5)
    if save_plots:
        plt.savefig(mean_plot_filename, format="png",
                   dpi=150,
                   bbox_inches="tight")
    plt.show()

    # Plot for the standard deviation model
    plt.figure(figsize=(8, 4))
    gr = updates_std[-1].reshape(cfg.params.nz, cfg.params.nx)
    g_min, g_max = np.percentile(gr, [2, 98])
    plt.imshow(
        gr, 
        cmap="Reds", 
        vmin=g_min, 
        vmax=g_max,
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    plt.title("std")
    plt.colorbar(shrink=0.5)
    if save_plots:
        plt.savefig(std_plot_filename, format="png",
                    dpi=150,
                    bbox_inches="tight")
    plt.show()
    
    # Plot for the kernel
    plt.figure(figsize=(8, 4))
    gr = kernel[-1]
    # g_min, g_max = np.percentile(gr, [2, 98])
    plt.imshow(
        gr, 
        cmap="Blues", 
        vmin=0, 
        vmax=1
    )
    plt.title("kernel")
    plt.colorbar(shrink=0.5)
    if save_plots:
        plt.savefig(kernel_plot_filename, format="png",
                    dpi=150,
                    bbox_inches="tight")
    plt.show()
    
    

def make_video_attribute(attribute, vmin=None, vmax=None, cmap='viridis', title=None, 
                         colorbar=True, fig_size=(6, 6), epochs=10, video_dir='../videos', 
                         nz=100, nx=150,
                         temp_dir='temp_frames_attributes', video_name='00_video_attribute.mp4', fps=15):

    """
    Plots a 2D density plot of an attribute over multiple frames and saves it as a video.

    Parameters:
    attribute (numpy.ndarray): An array of 3D attribute data where each frame represents a mean/std plot.
    vmin (float, optional): Minimum value for the colormap scale. Default is None.
    vmax (float, optional): Maximum value for the colormap scale. Default is None.
    cmap (str, optional): Colormap for the density plot. Default is 'viridis'.
    title (str, optional): Title for the plot. Default is None.
    colorbar (bool, optional): Whether to include a colorbar. Default is True.
    fig_size (tuple, optional): Figure size. Default is (6, 6).
    epochs (int, optional): Number of frames to create for the video. Default is 10.
    video_dir (str, optional): Directory to save the video. Default is '../videos'.
    video_name (str, optional): Name of the output video file. Default is '00_video_attribute.mp4'.
    fps (int, optional): Frames per second for the video. Default is 15.

    Returns:
    None: The function displays the plot but does not return any object.
    """

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)


    for i in tqdm(range(epochs), desc="Making Video"):

        plt.figure(figsize=fig_size)
        plt.imshow(attribute[i].reshape(nz, nx), cmap=cmap, vmin=vmin, vmax=vmax)
    
        if colorbar:
            plt.colorbar(shrink=0.5)
        if title is not None:
            plt.title(f'epoch: {i} - {title}')

        plt.tight_layout()

        # Save frame as image
        plt.savefig(os.path.join(temp_dir, f'frame_{i}.png')) #, bbox_inches='tight')
        plt.close()

    with imageio.get_writer(os.path.join(video_dir, video_name), fps=fps) as writer:
        for i in range(epochs):
            frame = imageio.imread(os.path.join(temp_dir, f'frame_{i}.png'))
            writer.append_data(frame)

    shutil.rmtree(temp_dir)
    
    return print('Video saved in:', video_dir)




def make_video_pca(pca_data, title=None, fig_size=(4, 4), epochs=10, 
                   video_dir='../videos', temp_dir='temp_frames_pca', 
                   video_name='00_video_pca.mp4', fps=15, marker_size=50):
    """
    Plots a density KDE and 2D scatter plot of PCA-reduced data over multiple frames with each particle having a unique color
    and saves it as a video, ensuring consistent x and y axis limits across all plots with added transparency
    and increased marker size.
    """

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate global x and y limits
    x_min = np.min(pca_data[:, :, 0])
    x_max = np.max(pca_data[:, :, 0])
    y_min = np.min(pca_data[:, :, 1])
    y_max = np.max(pca_data[:, :, 1])

    # Generate a unique color for each particle
    num_particles = pca_data.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, num_particles))

    for i in tqdm(range(epochs), desc="Making Video"):
        fig, ax = plt.subplots(figsize=fig_size)
        sns.kdeplot(x=pca_data[i, :, 0], y=pca_data[i, :, 1], fill=True)
        for j in range(num_particles):
            ax.scatter(pca_data[i, j, 0], pca_data[i, j, 1], color=colors[j], alpha=0.9, s=marker_size)
        
        # Apply consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if title:
            plt.title(f'{title} - Epoch: {i}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, f'frame_{i}.png'))
        plt.close()

    with imageio.get_writer(os.path.join(video_dir, video_name), fps=fps) as writer:
        for i in range(epochs):
            frame = imageio.imread(os.path.join(temp_dir, f'frame_{i}.png'))
            writer.append_data(frame)

    shutil.rmtree(temp_dir)
    print('Video saved in:', video_dir)
    
    
    

def plot_multiple_means_stds(global_mean, global_std, mean_cluster_0, std_cluster_0, mean_cluster_1, std_cluster_1, 
                             cfg, m_vmin, m_vmax, s_vmin, s_vmax, x, z, plot_path, save_plots=False):
    """
    Plots multiple mean and standard deviation models with consistent color scales.

    Parameters:
    - global_mean: Global mean array.
    - global_std: Global standard deviation array.
    - mean_cluster_0: Mean array for cluster 0.
    - std_cluster_0: Standard deviation array for cluster 0.
    - mean_cluster_1: Mean array for cluster 1.
    - std_cluster_1: Standard deviation array for cluster 1.
    - cfg: A configuration object with attributes `params.nz` and `params.nx` for model dimensions.
    - m_vmin, m_vmax: The minimum and maximum values for the color scale of the mean model plots.
    - s_vmin, s_vmax: The minimum and maximum values for the color scale of the std model plots.
    - x, z: Coordinate arrays for the horizontal and vertical axes, respectively.
    """

    # Create subplots with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 4), constrained_layout=True, sharey=True, sharex=True)

    # Plot global mean
    im1 = axes[0, 0].imshow(
        global_mean.reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[0, 0].set_title("Global")
    axes[0, 0].set_ylabel("Mean")
    fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # Plot cluster 0 mean
    im2 = axes[0, 1].imshow(
        mean_cluster_0.reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[0, 1].set_title("Cluster 0")
    fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # Plot cluster 1 mean
    im3 = axes[0, 2].imshow(
        mean_cluster_1.reshape(cfg.params.nz, cfg.params.nx),
        vmin=m_vmin,
        vmax=m_vmax,
        cmap="terrain",
        interpolation="bilinear",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[0, 2].set_title("Cluster 1")
    fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)

    # Plot global standard deviation
    im4 = axes[1, 0].imshow(
        global_std.reshape(cfg.params.nz, cfg.params.nx),
        vmin=s_vmin,
        vmax=s_vmax,
        cmap="Reds",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[1, 0].set_title("Global")
    axes[1, 0].set_ylabel("Std")
    fig.colorbar(im4, ax=axes[1, 0], shrink=0.8)

    # Plot cluster 0 standard deviation
    im5 = axes[1, 1].imshow(
        std_cluster_0.reshape(cfg.params.nz, cfg.params.nx),
        vmin=s_vmin,
        vmax=s_vmax,
        cmap="Reds",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[1, 1].set_title("Cluster 0")
    fig.colorbar(im5, ax=axes[1, 1], shrink=0.8)

    # Plot cluster 1 standard deviation
    im6 = axes[1, 2].imshow(
        std_cluster_1.reshape(cfg.params.nz, cfg.params.nx),
        vmin=s_vmin,
        vmax=s_vmax,
        cmap="Reds",
        extent=[x[0], x[-1], z[-1], z[0]]
    )
    axes[1, 2].set_title("Cluster 1")
    fig.colorbar(im6, ax=axes[1, 2], shrink=0.8)

    # plt.tight_layout()
    if save_plots is not False:
        plt.savefig(plot_path, format="png", dpi=150, bbox_inches="tight")
    plt.show()

