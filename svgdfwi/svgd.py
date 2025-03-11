import numpy as np
import torch
import torch.optim as optim
from svgdfwi.fwi import *
from torch.distributions.distribution import Distribution
from pylops import LinearOperator
import math
import gc


class SVGD:
    def __init__(self, particles, K, alpha, optimizer, scheduler, device='cuda'):
        """
        Simulate SVGD for n number of particles

        gmax: max FWI gradient of the initial particles (numpy.ndarray or torch.Tensor)
        K: Kernel function (e.g., RBF or IMQ) taking two arguments and returning a matrix
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler (optional)
        compute_gradient_per_batch: Function to compute FWI gradient per batches
        grad_func: Gradient of each FWI modeling
        filter_func: Function to filter the gradient
        nz: Model dimensions in z
        nx: Model dimensions in x
        device: The torch device on which computations will be performed
        """
        self.particles = particles
        self.K = K
        self.alpha = alpha
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    

        # Variables to retrieve later
        self.log_grad_p = None
        self.sigma = None
        self.K_XX = None
        self.gradients = None
        self.driving_force = None
        self.repulsive_force = None

    def phi(self, particles, log_grad_p, epoch, EMA):
        """
        Compute the Stein Gradient. The terms inside the square bracket above.

        X: n number of models (particles)
        """
        X = particles.detach().requires_grad_(True)
        X = particles.clone()
        
        self.log_grad_p = log_grad_p
        self.K_XX, der, self.sigma = self.K(X, EMA)
    
        self.driving_force = self.K_XX.mm(self.log_grad_p)
        self.repulsive_force = der
        
        alpha_iter = self.alpha[epoch]
        
        phi = alpha_iter*self.driving_force - self.repulsive_force
        
        return phi

    def step(self, X, log_grad_p, m_vmin, m_vmax, epoch, gmax=1, EMA=None):
        """
        Bound model to the limits
        m_vmin: minimum model value
        m_vmax: maximum model value
        """
        self.optimizer.zero_grad()
        X.grad = self.phi(X, log_grad_p, epoch, EMA)/gmax
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        with torch.no_grad():
            X.clamp_(m_vmin, m_vmax*1.15)
