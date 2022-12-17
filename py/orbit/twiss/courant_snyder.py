"""Courant-Snyder parameterization of uncoupled (transverse) motion.

References
----------
[1] 
"""
import numpy as np
from scipy.linalg import block_diag


def norm_matrix(*twiss_params):
    """Symplectic 2n x 2n normalization matrix.
        
    The matrix is block-diagonal. Each 2 x 2 block is defined by a set of
    Twiss parameters in a two-dimensional phase space (x-x', y-y', etc.). 
    Each set of Twiss parameters defines an ellipse; this matrix transforms
    each ellipse into a circle.
        
    Parameters
    ----------
    alpha_x, beta_x, alpha_y, beta_y, ... : float
        Twiss parameters for each dimension (x-x', y-y', ...).
        
    Returns
    -------
    V : ndarray, shape (2n, 2n)
        Block-diagonal normalization matrix. (2n is length of `twiss_params`.)
    """
    def norm_matrix_2x2(alpha, beta):
        return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)
    
    Vii = []
    for i in range(0, len(twiss_params), 2):
        alpha, beta = twiss_params[i : i + 2]
        Vii.append(norm_matrix_2x2(alpha, beta))
    return block_diag(*Vii)


def analyze_transfer_matrix_2x2(M):
    """Compute parameters from periodic 2 x 2 transfer matrix M.
        
    Parameters
    ----------
    M : ndarray, shape (2, 2)
        A u-u' transfer matrix.
    
    Returns
    -------
    dict
        'beta' : float
            The periodic beta function.
        'alpha' : float
            The periodic alpha function.
    """
    params = dict()
    cos_phi = (M[0, 0] + M[1, 1]) / 2.0
    sign = 1.0
    if abs(M[0, 1]) != 0:
        sign = M[0, 1] / abs(M[0, 1])
    sin_phi = sign * np.sqrt(1.0 - cos_phi**2)
    beta = M[0, 1] / sin_phi
    alpha = (M[0, 0] - M[1, 1]) / (2.0 * sin_phi)
    params['alpha'] = alpha
    params['beta'] = beta
    params['tune'] = np.arccos(cos_phi) / (2.0 * np.pi) * sign
    return params


def analyze_transfer_matrix(M):
    """Compute parameters from periodic, uncoupled 4 x 4 transfer matrix M.
    
    The tunes do not need to be computed in this method.
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        An x-x'-y-y' transfer matrix.
    
    Returns
    -------
    dict
        'beta_x' : float
            The periodic x beta function.
        'beta_y' : float
            The periodic y beta function.
        'alpha_x' : float
            The periodic x alpha function.
        'alpha_y' : float
            The periodic y alpha function.
    """
    params_x = analyze_transfer_matrix_2x2(M[0:2, 0:2])
    params_y = analyze_transfer_matrix_2x2(M[2:4, 2:4])
    params = dict()
    params['alpha_x'] = params_x['alpha']
    params['alpha_y'] = params_y['alpha']
    params['beta_x'] = params_x['beta']
    params['beta_y'] = params_y['beta']
    return params