"""
Module to compute beam statistics.
"""

import numpy as np
from numpy import linalg as la


def to_vec(Sigma):
    """Return array of 10 unique elements of covariance matrix."""
    return Sigma[np.triu_indices(4)]
    
    
def intrinsic_emittances(Sigma):
    """Return intrinsic emittances from covariance matrix."""
    # Get imaginary components of eigenvalues of S.U
    U = np.array([[0,1,0,0], [-1,0,0,0], [0,0,0,1], [0,0,-1,0]])
    eigvals = la.eigvals(np.matmul(Sigma, U)).imag
    # Keep positive values
    eigvals = eigvals[np.argwhere(eigvals >= 0).flatten()]
    # If one of the mode emittances is zero, both will be kept.
    # Remove the extra zero.
    if len(eigvals) > 2:
        eigvals = eigvals[:-1]
    # Return the largest emittance first
    e1, e2 = np.sort(eigvals)
    return e1, e2
    
    
def apparent_emittances(Sigma):
    ex = np.sqrt(la.det(Sigma[:2, :2]))
    ey = np.sqrt(la.det(Sigma[2:, 2:]))
    return ex, ey
        
    
def get_twiss(Sigma):
    """Return Twiss parameters and emittances from covariance matrix."""
    ex, ey = apparent_emittances(Sigma)
    bx = Sigma[0, 0] / ex
    by = Sigma[2, 2] / ey
    ax = -Sigma[0, 1] / ex
    ay = -Sigma[2, 3] / ey
    e1, e2 = intrinsic_emittances(Sigma)
    return np.array([ax, ay, bx, by, ex, ey, e1, e2])
    
    
class Stats:
    """Container for beam statistics."""
    def __init__(self, X):
        self.Sigma = np.cov(X.T)
        self.moments = to_vec(self.Sigma)
        self.twiss = get_twiss(self.Sigma)
