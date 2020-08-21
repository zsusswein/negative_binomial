''' This module contains functions necessary to fit a negative binomial
using the maximum likelihood estimator and some numerical analysis

@author: Zachary Susswein (based on original code by Peter Xenopoulos)
'''

## Libraries
import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom
from matplotlib.pyplot import plot, hist

## Functions
def nu_sum(vec_element, k):
    '''
    This function efficiently computes the gamma function term of the NB log lik
    by expanding the sum into a grid. Treats the gamma function as a logged
    factorial because the data must be integer values.
    
    @param vec_element: an element of the data vector
    @param k: the value of the dispersion parameter
    '''
    
    nu = np.arange(0, vec_element, 1)

    return np.sum(np.log(1 + nu / k))

def neg_log_lik(k, y_bar, vec, n):
    '''
    This function computes the negative log likelihood of the NB dist. using the
    MLE estimate of the mean, y_bar, and a set version of the dispersion parameter.
    
    This approach produces a biased estimate because it does not account for
    the use of the unbiased estimator of the sample mean (y_bar) in the place
    of the population mean.
    
    @param k: the dispersion parameter
    @param y_bar: the sample mean, an unbiased estimator of the population mean
    @param vec: the data vector
    @param n: the number of observations
    '''
    
    x = 0
    for i in range(n):
        x += nu_sum(vec[i], k)
        
    log_lik = (x / n) + y_bar * np.log(y_bar) - (y_bar + k) * np.log(1 + y_bar / k)
    
    return -log_lik

def plot_pdf(k_hat, y_bar, vec):
    '''
    plot the estimated pmf over the data
    
    @param k_hat: the estimated value of the NB dispersion parameter
    @param y_bar: the estimated value of the NB mean
    '''
    
    
    p_hat = (y_bar**2 / k_hat) / (y_bar + (y_bar**2 / k_hat))
    n_hat = y_bar**2 / (y_bar**2 / k_hat)
    
    x = np.arange(min(vec), max(vec + 1), 1)
    
    y_tilde = nbinom(n = n_hat,
                     p = p_hat)
    
    
    hist(vec, alpha = .2)
    plot(y_tilde.pmf(x) * len(vec), color = 'blue')
    
    return None
    

def neg_bin_fit(vec, init = 1, plot = False):
    '''
    Function to fit negative binomial dist. to data. Assumes that underdispersion
    does not occur, which guarantees the score has at least one root in the positive reals.
    
    Uses the mean and dispersion parameterization of the pmf common in ecology.
    
    @param vec: The data vector used to fit the negative binomial distribution
    @param init: The initial estimate for k, the dispersion parameter
    @param plot: whether to plot the fitted distribution over the data
    '''
    
    #####
    ## Type and data checking
    
    # Check the input is properly specified
    if not isinstance(vec, np.ndarray):
        raise TypeError("Argument 'vec' must be a numpy.ndarray")
    
    if len(vec.shape) != 1:
        raise TypeError("Argument 'vec' must be a vector with shape (n,)")
    
    if (not np.issubdtype(vec.dtype, np.integer)):
    	raise ValueError("Numpy array elements must be of type int")
    
    if type(plot) is not bool:
        raise TypeError('Argument `plot` must be a boolean')
    
    if (type(init) is not float) & (type(init) is not int):
        raise TypeError('Argument `init` must be of type float or type int')
    
    if init <= 0:
        raise ValueError('Argument `init` must be greater than zero')
    
    # Check the data
    if np.sum(vec < 0) > 0:
        raise ValueError("Data must all be greater than or equal to zero, negative number provided")
    
    if np.mean(vec) > np.var(vec):
        raise ValueError("Data are underdispersed; fitting method does not allow for underdispersion")
    
    #####
    ## Fit the NB dist. to the vector
    
    # MLE of the mean
    y_bar = np.mean(vec)
    
    # MLE of k
    fit = minimize(fun = neg_log_lik, 
               x0 = 1, 
               args = (np.mean(vec), vec, len(vec),), 
               method = 'L-BFGS-B', 
               bounds = ((0.00001, None),))


    if plot:
        
        plot_pdf(fit['x'][0], y_bar, vec)

    return y_bar, fit['x'][0]











