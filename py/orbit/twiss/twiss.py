"""Analysis of linear, periodic transfer maps."""
import numpy as np
from scipy.linalg import block_diag

import orbit.twiss.courant_snyder as CS
import orbit.twiss.edwards_teng as ET
import orbit.twiss.lebedev_bogacz as LB
import orbit.twiss.qin_davidson as QD


def unit_symplectic_matrix(n=2):
    """Construct 2n x 2n unit symplectic matrix.

    Each 2 x 2 block is [[0, 1], [-1, 0]]. This assumes our phase space vector
    is ordered as [x, x', y, y', ...].
    """
    if n % 2 != 0:
        raise ValueError("n must be an even integer")
    U = np.zeros((n, n))
    for i in range(0, n, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def is_symplectic(M, tol=1.0e-6):
    """Return True if M is symplectic.
    
    M is symplectic iff M^T * U * M = U, where U is the unit symplectic matrix.
    This method checks whether the sum of squares of U - M^T * U * M is less 
    than `tol`.
    """
    if M.shape[0] != M.shape[1]:
        return False
    U = unit_symplectic_matrix(M.shape[0])
    residuals = U - np.linalg.multi_dot([M.T, U, M])
    return np.sum(residuals**2) < tol


def is_coupled(M):
    """Return True if there are coupled elements in M."""
    if M.shape[0] < 4:
        return False
    mask = np.zeros(M.shape)
    for i in range(0, M.shape[0], 2):
        mask[i : i + 2, i : i + 2] = 1.0
    return np.any(np.ma.masked_array(M, mask=mask))


def is_stable(M, tol=1.0e-8):
    """Return True if M produces bounded motion."""
    return all_eigvals_on_unit_circle(np.linalg.eigvals(M), tol=tol)

    
def all_eigvals_on_unit_circle(eigvals, tol=1.0e-8):
    """Return True if all eigenvalues are on the unit circle in the complex plane.
    
    eigvals : ndarray, shape (n,)
        Eigenvalues of a symplectic transfer matrix.
    """
    for eigval in eigvals:
        if abs(np.linalg.norm(eigval) - 1.0) > tol:
            return False
    return True


def eigtunes_from_eigvals(eigvals):
    """Return eigentune from eigenvalue of symplectic matrix.
    
    They are related as: eigval = Re[exp(-i * (2 * pi * tune))], where i is the imaginary unit.
    
    eigvals : ndarray, shape (n,)
        Eigenvalues of a symplectic transfer matrix.
    """
    return np.arccos(eigvals.real)[[0, 2]] / (2.0 * np.pi)


def phase_adv_matrix(*phase_advances):
    """2n x 2n phase advance matrix (clockwise rotation in each phase plane).
    
    Parameters
    ---------
    mu1, mu2, ..., mun : float
        The phase advance in each plane.
    
    Returns
    -------
    ndarray, shape (2n, 2n)
        Matrix that rotates x-x', y-y', z-z', etc.
    """
    def rotation_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, s], [-s, c]])
    
    mats = [rotation_matrix(phase_advance) for phase_advance in phase_advances]
    return block_diag(*mats)


def construct_transfer_matrix(norm_matrix, phase_adv_matrix):
    """Construct transfer matrix from normalization and phase advance matrix."""
    V = norm_matrix
    P = phase_adv_matrix
    Vinv = np.linalg.inv(V)
    return np.linalg.multi_dot([V, P, Vinv])


class TransferMatrix:
    """Symplectic transfer matrix analysis.
    
    Attributes
    -----------
    M : ndarray, shape (2n, 2n)
        The periodic transfer matrix.
    eigvals : ndarray, shape (2n,)
        The eigenvalues of the transfer matrix.
    eigvecs : ndarray, shape (2n, 2n)
        The eigenvectors of the transfer matrix. (Arranged as columns.)
    eigtunes : ndarray, shape (n,)
        The tunes {nu_1, nu_2, ... nun}. The tune nu_l is related to the 
        eigenvalue lambda_l as: lambda_l = Re[exp(-i * (2 * pi * nu_l))],
        where i is the imaginary unit.
    stable : bool
        Whether the transfer matrix is stable --- whether all eigenvalues
        lie on the unit circle in the complex plane.
    coupled : bool
        Whether there are any nonzero off-block-diagonal (cross-plane) elements.
    parameterization : {'CS', 'LB', 'ET', 'QD'}
        * 'CS': Courant-Snyder
        * 'LB': Lebedev-Bogacz 
        * 'ET': 'Edwards-Teng 
        * 'QD': Qin-Davidson
    parameters : dict
        Computed lattice parameters. Each parameterization above will return
        different parameters; see each module/paper for details.
    """
    def __init__(self, M, parameterization='CS'):
        self.M = M
        self.eigvecs = None
        self.eigvals = None
        self.eigtunes = None
        self.stable = False
        self.parameterization = parameterization            
        self.params = dict()
        self.analyze_eig()
        self.analyze()
        
    def set_parameterization(self, parameterization):
        if parameterization not in ['CS', 'LB']:
            raise ValueError('Invalid parameterization.')
        self.parameterization = parameterization
        
    def analyze_eig(self):
        self.eigvals, self.eigvecs = np.linalg.eig(self.M)
        self.eigtunes = eigtunes_from_eigvals(self.eigvals)
        self.stable = all_eigvals_on_unit_circle(self.eigvals)
        self.coupled = is_coupled(self.M)
        
    def analyze(self):
        if self.parameterization == 'CS':
            self.params = CS.analyze_transfer_matrix(self.M)
        elif self.parameterization == 'LB':
            self.params = LB.analyze_transfer_matrix(self.M)