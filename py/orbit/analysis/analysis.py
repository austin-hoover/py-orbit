"""Functions to compute the statistical beam parameters."""

import numpy as np
from numpy import linalg as la


# Symplectic matrix for mode emittance calculation
U = np.array([[0,1,0,0], [-1,0,0,0], [0,0,0,1], [0,0,-1,0]])


def coords(bunch):
    """Return the coordinate array from bunch."""
    nparts = bunch.getSize()
    X = np.zeros((nparts, 4))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
    return X


def mode_emittances(S):
    """Return the mode emittances from the covariance matrix."""
    
    # Get imaginary components of eigenvalues of S.U
    eigvals = la.eigvals(np.matmul(S, U)).imag

    # Keep positive values
    eigvals = eigvals[np.argwhere(eigvals >= 0).flatten()]

    # If one of the mode emittances is zero, both will be kept.
    # Remove the extra zero.
    if len(eigvals) > 2:
        eigvals = eigvals[:-1]
        
    # Return the largest eigenvalue first
    eigvals = np.sort(eigvals)
    e1, e2 = np.sort(eigvals)
    return e1, e2


def twiss(S):
    """Compute the transverse Twiss parameters.

    Parameters
    ----------
    S : NumPy array, shape (4, 4) 
        The transverse covariance matrix.
        
    Returns
    -------
    ax{y} : float
        The x{y} alpha parameter.
    bx{y} : float
        The x{y} beta parameter.
    ex{y} : float
        The x{y} rms emittance.
    e1{2} : float
        The mode emittance.
    """
    ex = np.sqrt(la.det(S[:2, :2]))
    ey = np.sqrt(la.det(S[2:, 2:]))
    bx = S[0, 0] / ex
    by = S[2, 2] / ey
    ax = -S[0, 1] / ex
    ay = -S[2, 3] / ey
    e1, e2 = mode_emittances(S)
    return ax, ay, bx, by, ex, ey, e1, e2


class Stats:
    
    def __init__(self, file_path):
        self.file_twiss = open(''.join([file_path, '/twiss.dat']), 'a')
        self.file_moments = open(''.join([file_path, '/moments.dat']), 'a')

    def write(self, s, bunch):
        X = 1000 * coords(bunch) # mm-mrad
        S = np.cov(X.T)
        ax, ay, bx, by, ex, ey, e1, e2 = twiss(S)
        f1 = '{:.4f}' + 8*' {:.5f}' + '\n'
        f2 = '{:.4f}' + 10*' {:.5f}' + '\n'
        self.file_twiss.write(f1.format(s, ax, ay, bx, by, ex, ey, e1, e2))
        self.file_moments.write(f2.format(s, S[0,0], S[0,1], S[0,2], S[0,3], S[1,1], S[1,2], S[1,3], S[2,2], S[2,3], S[3,3]))
        
    def close(self):
        self.file_twiss.close()
        self.file_moments.close()
