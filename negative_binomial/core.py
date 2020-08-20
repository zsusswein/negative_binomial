''' This module contains functions necessary to fit a negative binomial
using the maximum likelihood estimator and some numerical analysis

@author: Peter Xenopoulos
@website: http://www.peterxeno.com
'''

import math
import numpy as np

from scipy.optimize import newton
from scipy.special import digamma

def r_derv(r_var, vec):
    ''' Function that represents the derivative of the neg bin likelihood wrt r
    @param r: The value of r in the derivative of the likelihood wrt r
    @param vec: The data vector used in the likelihood
    '''

    total_sum = 0
    obs_mean = np.mean(vec)  # Save the mean of the data
    n_pop = float(len(vec))  # Save the length of the vector, n_pop

    for obs in vec:
        total_sum += digamma(obs + r_var)

    total_sum -= n_pop*digamma(r_var)
    total_sum += n_pop*math.log(r_var / (r_var + obs_mean))

    return total_sum

def p_equa(r_var, vec):
    ''' Function that represents the equation for p in the neg bin likelihood wrt p
    @param r: The value of r in the derivative of the likelihood wrt p
    @param vec: Te data vector used in the likelihood
    '''

    data_sum = np.sum(vec)
    n_pop = float(len(vec))
    p_var = 1 - (data_sum / (n_pop * r_var + data_sum))
    return p_var


def dl_dmu(mu, vec):
    ''' 
    Function represents the partial derivative of the log-likelihood with respect
    to the mean of the negative binomial distribution.
    
    @param vec: The data vector
    '''
    
    dl_dmu = np.sum(np.divide(vec, mu) - np.divide(1 + (vec / k), 1 + (mu / k)))

def neg_bin_fit(vec, init=0.0001):
    '''
    Function to fit negative binomial to data. Assumes that underdispersion
    does not occur.
    @param vec: The data vector used to fit the negative binomial distribution
    @param init: Set init to a number close to 0, and you will always converge
    '''
    
    #####
    ## Type and data checking
    
    # Check the input is properly specified
    if not isinstance(vec, np.ndarray):
        raise TypeError("Argument 'vec' must be a numpy.ndarray")
    
    if len(vec.shape) != 1:
        raise TypeError("Argument 'vec' must be a vector with shape (n,)")
    
    if (not np.issubdtype(vec.dtype, np.floating)) & (not np.issubdtype(vec.dtype, np.integer)):
    	raise ValueError("Numpy array elements must be of type float or type int")
    
    # Check the data
    if np.sum(vec < 0) > 0:
        raise ValueError("Data must all be greater than or equal to zero, negative number provided")
    
    if np.mean(vec) > np.var(vec):
        raise ValueError("Data are underdispersed; fitting method does not allow for underdispersion")
    
    #####
    ## Fit the NB dist. to the vector
    
    est_r = newton(r_derv, init, args=(vec,))
    est_p = p_equa(est_r, vec)
    
    return est_r, est_p




