import numpy as np
from scipy.linalg import block_diag

import parameterizations.courant_snyder as CS
import parameterizations.lebedev_bogacz as LB
import parameterizations.edwards_teng as ET


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
    return all_eigenvalues_on_unit_circle(np.linalg.eigenvalues(M), tol=tol)


def all_eigenvalues_on_unit_circle(eigenvalues, tol=1.0e-8):
    """Return True if all eigenvalues are on the unit circle in the complex plane.

    eigenvalues : ndarray, shape (n,)
        Eigenvalues of a symplectic transfer matrix.
    """
    for eigval in eigenvalues:
        if abs(np.linalg.norm(eigval) - 1.0) > tol:
            return False
    return True


def eigentunes_from_eigenvalues(eigenvalues):
    """Return eigentunes from eigenvalues of symplectic matrix.

    They are related as: eigval = exp(-i * (2 * pi * tune)), where i is the imaginary unit.

    Parameters
    ----------
    eigenvalues : ndarray, shape (2n,)
        Eigenvalues of a symplectic transfer matrix.

    Returns
    -------
    ndarray, shape (n,)
        The eigentunes.
    """

    def eigtune_from_eigval(eigval):
        return np.arccos(eigval.real) / (2.0 * np.pi)

    return np.array([eigtune_from_eigval(eigenvalues[k]) for k in (0, 2)])


def phase_advance_matrix(*phase_advances):
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


def normal_form(M, V):
    return np.linalg.multi_dot([np.linalg.inv(V), M, V])


def construct_transfer_matrix(normalization_matrix, phase_advance_matrix):
    """Construct transfer matrix from normalization and phase advance matrix."""
    V = normalization_matrix
    Vinv = np.linalg.inv(V)
    return np.linalg.multi_dot([V, phase_advance_matrix, Vinv])


class TransferMatrix:
    """Symplectic transfer matrix analysis.

    Attributes
    -----------
    M : ndarray, shape (2n, 2n)
        The periodic transfer matrix.
    eigenvalues : ndarray, shape (2n,)
        The eigenvalues of the transfer matrix.
    eigenvectors : ndarray, shape (2n, 2n)
        The eigenvectors of the transfer matrix. (Arranged as columns.)
    eigentunes : ndarray, shape (n,)
        The tunes {nu_1, nu_2, ... nun}. The tune nu_l is related to the
        eigenvalue lambda_l as: lambda_l = Re[exp(-i * (2 * pi * nu_l))],
        where i is the imaginary unit.
    stable : bool
        Whether the transfer matrix is stable --- whether all eigenvalues
        lie on the unit circle in the complex plane.
    coupled : bool
        Whether there are any nonzero off-block-diagonal (cross-plane) elements.
    parameters : dict
        Computed lattice parameters. Each parameterization above will return
        different parameters; see each module/paper for details.
    """

    def __init__(self, M):
        self.M = M[:4, :4]
        self.eigenvectors = None
        self.eigenvalues = None
        self.eigentunes = None
        self.stable = False
        self.params = dict()
        self.analyze_eig()

    def analyze_eig(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)
        self.eigentunes = eigentunes_from_eigenvalues(self.eigenvalues)
        self.stable = all_eigenvalues_on_unit_circle(self.eigenvalues)
        self.coupled = is_coupled(self.M)

    def analyze(self):
        return

    def matched_cov(self, *intrinsic_emittances):
        V = self.params["V"]
        Sigma_n = np.diag(np.repeat(intrinsic_emittances, 2))
        return np.linalg.multi_dot([V, Sigma_n, V.T])


class TransferMatrixCourantSnyder(TransferMatrix):
    """Courant-Snyder parameterization of uncoupled (transverse) motion."""

    def __init__(self, M):
        TransferMatrix.__init__(self, M)
        self.analyze()

    def analyze(self):
        self.eigenvectors = CS.normalize(self.eigenvectors)
        self.params.update(**CS.twiss_from_transfer_matrix(self.M))
        self.params["V"] = CS.normalization_matrix_from_twiss(
            self.params["alpha_x"],
            self.params["beta_x"],
            self.params["alpha_y"],
            self.params["beta_y"],
        )


class TransferMatrixLebedevBogacz(TransferMatrix):
    """Lebedev-Bogacz parameterization of coupled motion.

    Under developement. Some of these methods have not been tested in a while.

    References
    ----------
    [1] https://iopscience.iop.org/article/10.1088/1748-0221/5/10/P10010
    """

    def __init__(self, M):
        TransferMatrix.__init__(self, M)
        self.analyze()

    def normalization_matrix_from_eigenvectors(self, eigenvectors, norm=False):
        """Construct symplectic normalization matrix.

        eigenvectors: ndarray, shape (2n, 2n)
            Each column is an eigenvector. We assume they are not normalized.
        norm : bool
            If False, assume the eigenvectors are already normalized.
        """
        if norm:
            eigenvectors = self.normalize(eigenvectors.copy())
        V = np.zeros(eigenvectors.shape)
        for i in range(0, V.shape[1], 2):
            V[:, i] = eigenvectors[:, i].real
            V[:, i + 1] = (1.0j * eigenvectors[:, i]).real
        return V

    def analyze(self):
        self.eigenvectors = LB.normalize(self.eigenvectors)
        self.params["V"] = LB.normalization_matrix_from_eigenvectors(self.eigenvectors)
        self.params.update(**LB.twiss_from_normalization_matrix(self.params["V"]))


class TransferMatrixEdwardsTeng(TransferMatrix):
    def __init__(self, M):
        TransferMatrix.__init__(self, M)
        self.analyze()

    def analyze(self):
        raise NotImplementedError
