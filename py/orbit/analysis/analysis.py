"""
Module to compute beam statistics.
"""

import numpy as np
from numpy import linalg as la


def covmat2vec(S):
    """Return array of 10 unique elements of covariance matrix."""
    return np.array([S[0,0], S[0,1], S[0,2], S[0,3], S[1,1], S[1,2], S[1,3],
                     S[2,2], S[2,3], S[3,3]])
    
    
def rms_ellipse_params(Sigma):
    s11, s33, s13 = Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]
    phi = 0.5 * np.arctan2(2 * s13, s11 - s33)
    cx = np.sqrt(2) * np.sqrt(s11 + s33 + np.sqrt((s11 - s33)**2 + 4*s13**2))
    cy = np.sqrt(2) * np.sqrt(s11 + s33 - np.sqrt((s11 - s33)**2 + 4*s13**2))
    return phi, cx, cy
    
    
def mode_emittances(Sigma):
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
    
    
def twiss(Sigma):
    ex = np.sqrt(la.det(Sigma[:2, :2]))
    ey = np.sqrt(la.det(Sigma[2:, 2:]))
    bx = Sigma[0, 0] / ex
    by = Sigma[2, 2] / ey
    ax = -Sigma[0, 1] / ex
    ay = -Sigma[2, 3] / ey
    e1, e2 = mode_emittances(Sigma)
    return np.array([ax, ay, bx, by, ex, ey, e1, e2])
    
    
class Stats:
    """Container for the beam statistics."""
    def __init__(self, X):
        Sigma = np.cov(X.T)
        self.moments = covmat2vec(Sigma)
        self.twiss = twiss(Sigma)
