import torch
import numpy as np
from torch import jit

def concordance_corr_numpy(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

@jit.script
def concordance_corr(x, y):
    '''Lin's Concordance Correlation Coefficient for PyTorch tensors'''
    mx = torch.mean(x)
    my = torch.mean(y)
    sxy = torch.sum((x - mx) * (y - my)) / x.shape[0]
    rhoc = 2*sxy / (torch.var(x,unbiased=False) + torch.var(y,unbiased=False) + (mx - my) ** 2)
    return rhoc

@jit.script
def concordance_corrcoef(x):
    """
    Compute Lin's Concordance Correlation Coefficient (CCC) between all pairs of rows.

    Arguments
    ---------
    x : 2D torch.Tensor [items x features]
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    """
    # Mean and variance per row
    mean_x = torch.mean(x, dim=1)
    var_x = torch.var(x, unbiased=False, dim=1)

    # Calculate covariance matrix of rows
    xm = x.sub(mean_x.unsqueeze(1))
    cov = xm @ xm.t() / (x.size(1))

    # Prepare for broadcasting (make it batch_size x 1)
    mean_x = mean_x.view(-1, 1)
    var_x = var_x.view(-1, 1)

    # Compute CCC according to the formula
    ccc = 2 * cov / (var_x + var_x.t() + (mean_x - mean_x.t())**2)

    return ccc

@jit.script
def weighted_concordance_corrcoef(data, weights=None, ddof: int = 0):
    '''Compute Weighted Concordance Correlation Coefficient (CCC) for given data.
       Assumes variables in rows and observations in columns.
       
       Parameters
       ----------
       data : 2D torch.Tensor or numpy.array
           Input data organized as numObservations X numFeatures.
       weights : 1D torch.Tensor or numpy.array, optional
           Weights for each observation. It should be of length numObservations.
           If not provided, all observations are given equal weight.
       ddof : int, optional
           Delta Degrees of Freedom for the variance calculation. The divisor used in
           calculations is N - ddof, where N represents the number of observations.
       
       Returns
       -------
       ccc : 2D torch.Tensor or numpy.array
           Pairwise weighted CCC matrix. If data is of size (n, m), return value will be of size (n, n).
           
       Note
       ----
       This function assumes that the input data are organized with rows as observations 
       and columns as features (numObservations X numFeatures).
    '''        
    if isinstance(data, torch.Tensor):
        device = data.device
        if weights is None: weights = torch.ones(data.shape[1], dtype=torch.float, device=data.device)
        ddof = torch.tensor(ddof, dtype=torch.float, requires_grad=False).to(data.device)
        weights_sum = weights.sum(0)
        mean = (data.t() * weights.unsqueeze(1)).sum(dim=0) / weights_sum
        demeaned = data.t() - mean
        var_ = ((demeaned ** 2) * weights.unsqueeze(1)).sum(dim=0) / (weights_sum - ddof.item())
        weighted_demeaned = demeaned * weights.sqrt().unsqueeze(1)
        cov_ = torch.mm(weighted_demeaned.t(), weighted_demeaned) / (weights_sum - ddof.item())                
        ccc = 2 * cov_ / (var_.unsqueeze(0) + var_.unsqueeze(1) + (mean.unsqueeze(0) - mean.unsqueeze(1))**2)
        
    elif isinstance(data, np.ndarray):
        if weights is None: weights = np.ones(data.shape[1])
        ddof = np.array(ddof)
        weights_sum = weights.sum()
        mean = (data.T * weights[:, np.newaxis]).sum(axis=0) / weights_sum
        demeaned = data.T - mean
        var_ = ((demeaned ** 2) * weights[:, np.newaxis]).sum(axis=0) / (weights_sum - ddof.item())
        cov_ = np.dot(demeaned.T, (demeaned * weights[:, np.newaxis])) / (weights_sum - ddof.item())
        ccc = 2 * cov_ / (var_[np.newaxis, :] + var_[:, np.newaxis] + (mean[np.newaxis, :] - mean[:, np.newaxis])**2)
        
    else:
        raise TypeError("Input data must be a torch.Tensor or a numpy.array.")
        
    return ccc

def weighted_concordance_corr(data, weights=None, ddof=0):
    return weighted_concordance_corrcoef(data, weights, ddof)[0,1]

def activation_weighted_concordance_corrcoef(data):
    num_items = data.shape[0]
    
    M = torch.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            data_ = data.index_select(dim=0, index=torch.tensor([i,j]))
            
            # might be more sensible weighting schemes:
            weights = data_.sum(dim=0)
            weights = weights - weights.min()
            M[i,j] = weighted_concordance_corr(data_, weights)
            
    return M