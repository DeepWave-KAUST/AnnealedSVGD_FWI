![LOGO](https://github.com/DeepWave-KAUST/AnnealedSVGD_FWI-dev/blob/main/asset/logo.png)

## Reproducible material for Annealed Stein Variational Gradient Descent for Improved Uncertainty Estimation in Full-Waveform Inversion
 <br />

Corrales M.<sup>1+</sup>, Berti S.<sup>2+</sup>, Denel B.<sup>3</sup>, Williamson P.<sup>3</sup>, Aleardi M.<sup>2</sup>, and Ravasi M.<sup>1</sup>  <br />
<sup>1</sup> King Abdullah University of Science and Technology (KAUST)  <br />
<sup>2</sup> University of Pisa  <br />
<sup>3</sup> TotalEnergies  <br />
<sup>+</sup> Equal contribution  <br />



## Project structure
This repository is organized as follows:

* :open_file_folder: **svgdfwi**: python library containing routines for SVGD-FWI.
* :open_file_folder: **data**: folder containing data.
* :open_file_folder: **notebooks**: set of jupyter notebooks testing the scripts.
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments (exp-for single frequency, and multiscale-for multiscale frequency FWI).
* :open_file_folder: **videos**: Videos and plots for experiment monitoring.
* :open_file_folder: **visualization**: python routines for final plots.

## Notebooks
The following notebooks are provided:

- :orange_book: ``00_testing_svgd_fwi.ipynb``: notebook performing the svgd setup for fwi;
- :orange_book: ``01_HDBSCAN_clustering.ipynb``: notebook performing HDBSCAN clustering in the particles.


## Scripts
Set of scripts that are detailed on the file :green_book: ``experiment_list.xlsx``.

## Getting started :alien: :flying_saucer: :cow2:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

> [!TIP]
> In case you are experiencing slow installation on your conda, we advise updating your conda with the following commands:
> 
> ```sh
> conda info
> conda update -n base conda
> conda install -n base conda-libmamba-solver
> conda config --set solver libmamba
> ```
> 
> This update will allow your conda to install packages in parallel.

> [!IMPORTANT]
> To install the env and package, simply run:
> ```sh
> ./install_env.sh
> ```
> It will take some time. If at the end you see the word `Done!` on your terminal, you are ready to go. After that, you can simply install your package (svgdfwi has to be activated):
> ```sh
> pip install .
> ```
> or in developer mode:
> ```sh
> pip install -e .
> ```
> 
> Remember to always activate the environment by typing:
> ```sh
> conda activate svgdfwi
> ```
> 
> Finally, to run the scripts, go to the scripts folder and run:
> ```sh
> nohup python -u $your_python_file$.py hydra.job.chdir=False > $your_python_file$.log &
> ```


> [!NOTE]  
> All experiments have been carried on a Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz equipped with a single NVIDIA A100-SXM4-80GB. Different environment 
configurations may be required for different combinations of workstation and GPU.
