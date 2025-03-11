import numpy as np
import torch

class annealed_tanh():
    def alpha(self, p, t, T):
        al = np.tanh((1.3*t/T)**p)
        return al

class cyclic():
    def alpha(self, p, t ,T, C):
        al=((np.mod(t, T/C))/(T/C))**p
        return al
    
    
def alpha_tanh(total_epochs, p=2, device='cpu'):
    """
    Generate an alpha array using a tanh function across epochs.

    Parameters:
    total_epochs (int): Total number of epochs for the simulation.
    p (float): Power parameter for the alpha calculation. Default is 1.
    device (str): The torch device on which the alpha array will be generated. Default is 'cpu'.
    
    Using a factor of 2.0 to allow more epochs with alpha = 1.0
    Returns:
    torch.Tensor: An array of alpha values for each epoch.
    """
    t = torch.arange(1, total_epochs + 1, device=device)  # Epochs start from 1 to total_epochs
    T = total_epochs
    alpha = torch.tanh((2.0 * (t.float() / T)) ** p)
    
    return alpha

def alpha_tanh_cutoff(total_epochs, p=2, cutoff=0.8, device='cpu'):
    """
    Generate an alpha array using a tanh function across the first 80% of epochs, 
    setting alpha to 1 for the remaining 20%.

    Parameters:
        total_epochs (int): Total number of epochs for the simulation.
        p (float): Power parameter for the alpha calculation. Default is 2.
        cutoff (float): percentage of epochs where alpha is calculated using the tanh function.
        device (str): The torch device on which the alpha array will be generated. Default is 'cpu'.
    
    Returns:
        torch.Tensor: An array of alpha values for each epoch, where alpha is determined by a 
        tanh function for the first 80% of epochs and set to 1 thereafter.
    """
    t = torch.arange(1, total_epochs + 1, device=device)  # Epoch indices
    T = total_epochs
    cutoff = int(cutoff * T)  # Compute the epoch number where alpha should start being 1
    
    # Alpha values calculated using tanh function for the first 80% of epochs
    alpha = torch.tanh((2.0 * (t[:cutoff].float() / cutoff)) ** p)
    
    # Append 1s for the remaining 20% of epochs
    remaining_alpha = torch.ones(total_epochs - cutoff, device=device)
    alpha = torch.cat((alpha, remaining_alpha), dim=0)
    
    return alpha


def alpha_cyclic(T, C=4, p=3, device='cpu'):
    """
    Generate an alpha array based on the specified parameters.
    It has to be a minimum of 4 cycles
    Parameters:
    T (int): Total number of epochs for the simulation.
    C (int): Number of cycles for modulation of the alpha value.
    p (float): Exponent determining the speed of the transition between phases.
    device (str): The torch device on which the alpha array will be generated.

    Returns:
    torch.Tensor: An array of alpha values for each epoch.
    """
    # Initialize the alpha tensor on the specified device
    alpha = torch.zeros(T, device=device)
    
    # Calculate the length of each cycle, ensuring it's at least 1 to avoid division by zero
    cycle_length = max(torch.div(T, C, rounding_mode='floor').item(), 1)
    
    for i in range(1, T + 1):  # Looping from 1 to T to match the intended logic
        # Calculate the current epoch's alpha value based on its position in the cycle
        position = i % cycle_length if i % cycle_length != 0 else cycle_length
        alpha[i - 1] = (position / cycle_length) ** p
        
        # Check if the current epoch is in the last 2 cycles and set alpha to 1.0 if so
        if i > (T // C) * (C - 2):
            alpha[i - 1] = 1.0
         
    return alpha