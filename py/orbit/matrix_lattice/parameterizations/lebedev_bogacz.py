"""Lebedev-Bogacz parameterization of linear coupled motion.

Work in progress.
"""
import numpy as np
from scipy.linalg import block_diag


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


def normalize(eigvecs):
    """Normalize transfer matrix eigenvectors.

    eigvecs: ndarray, shape (2n, 2n)
        Each column is an eigenvector.
    """
    n = eigvecs.shape[0]
    U = unit_symplectic_matrix(n)
    for i in range(0, n, 2):
        v = eigvecs[:, i]
        # Find out if we need to swap.
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0:
            eigvecs[:, i], eigvecs[:, i + 1] = eigvecs[:, i + 1], eigvecs[:, i]
        eigvecs[:, i : i + 2] *= np.sqrt(2 / np.abs(val))
    return eigvecs


def normalization_matrix_from_eigvecs(eigvecs, norm=True):
    """Construct symplectic normalization matrix.

    eigvecs: ndarray, shape (2n, 2n)
        Each column is an eigenvector. We assume they are not normalized.
    norm : bool
        If False, assume the eigenvectors are already normalized.
    """
    if norm:
        eigvecs = normalize(eigvecs.copy())
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1j * eigvecs[:, i]).real
    return V


def normalization_matrix_from_twiss_one_mode(
    alpha_lx=None, 
    alpha_ly=None, 
    beta_lx=None, 
    beta_ly=None, 
    nu=None,
    u=None,
    mode=1,
):
    _cos = np.cos(nu)
    _sin = np.sin(nu)
    V = np.zeros((4, 4))
    if mode == 1:
        V[0, 0] = np.sqrt(beta_lx)
        V[0, 1] = 0.0
        V[1, 0] = -alpha_lx / np.sqrt(beta_lx)
        V[1, 1] = (1.0 - u) / np.sqrt(beta_lx)
        V[2, 0] = +np.sqrt(beta_ly) * _cos
        V[2, 1] = -np.sqrt(beta_ly) * _sin
        V[3, 0] = (u * _sin - alpha_ly * _cos) / np.sqrt(beta_ly)
        V[3, 1] = (u * _cos + alpha_ly * _sin) / np.sqrt(beta_ly)
    elif mode == 2:
        V[0, 2] = +np.sqrt(beta_lx) * _cos
        V[0, 3] = -np.sqrt(beta_lx) * _sin
        V[1, 2] = (u * _sin - alpha_lx * _cos) / np.sqrt(beta_lx)
        V[1, 3] = (u * _cos + alpha_lx * _sin) / np.sqrt(beta_lx)
        V[2, 2] = np.sqrt(beta_ly)
        V[2, 3] = 0.0
        V[3, 2] = -alpha_ly / np.sqrt(beta_ly)
        V[3, 3] = (1.0 - u) / np.sqrt(beta_ly)
    return V


def twiss_from_normalization_matrix(V):
    beta_1x = V[0, 0] ** 2
    beta_2y = V[2, 2] ** 2
    alpha_1x = -np.sqrt(beta_1x) * V[1, 0]
    alpha_2y = -np.sqrt(beta_2y) * V[3, 2]
    u = 1.0 - (V[0, 0] * V[1, 1])
    nu1 = np.arctan2(-V[2, 1], V[2, 0])
    nu2 = np.arctan2(-V[0, 3], V[0, 2])
    beta_1y = (V[2, 0] / np.cos(nu1)) ** 2
    beta_2x = (V[0, 2] / np.cos(nu2)) ** 2
    alpha_1y = (u * np.sin(nu1) - V[3, 0] * np.sqrt(beta_1y)) / np.cos(nu1)
    alpha_2x = (u * np.sin(nu2) - V[1, 2] * np.sqrt(beta_2x)) / np.cos(nu2)
    return {
        "alpha_1x": alpha_1x,
        "alpha_1y": alpha_1y,
        "alpha_2x": alpha_2x,
        "alpha_2y": alpha_2y,
        "beta_1x": beta_1x,
        "beta_1y": beta_1y,
        "beta_2x": beta_2x,
        "beta_2y": beta_2y,
        "u": u,
        "nu1": nu1,
        "nu2": nu2,
    }


def matched_cov(M, *intrinsic_emittances):
    eigvals, eigvecs = np.linalg.eig(M)
    V = normalization_matrix_from_eigvecs(eigvecs)
    Sigma_n = np.diag(np.repeat(intrinsic_emittances, 2))
    return np.linalg.multi_dot([V, Sigma_n, V.T])


def symplectic_diag(self, Sigma):
    U = unit_symplectic_matrix(4)
    eigvals, eigvecs = np.linalg.eig(np.matmul(Sigma, U))
    V = normalization_matrix_from_eigvecs(eigvecs)
    Vinv = np.linalg.inv(V)
    return np.linalg.multi_dot([Vinv, Sigma, Vinv.T])


def cov_from_twiss_one_mode(
    alpha_lx=None, 
    alpha_ly=None, 
    beta_lx=None, 
    beta_ly=None, 
    u=None, 
    nu=None,
    eps=None, 
    mode=1
):
    cos, sin = np.cos(nu), np.sin(nu)
    if mode == 1:
        s11 = beta_lx
        s33 = beta_ly
        s12 = -alpha_lx
        s34 = -alpha_ly
        s22 = ((1.0 - u) ** 2 + alpha_lx**2) / beta_lx
        s44 = (u**2 + alpha_ly**2) / beta_ly
        s13 = np.sqrt(beta_lx * beta_ly) * cos
        s14 = np.sqrt(beta_lx / beta_ly) * (u * sin - alpha_ly * cos)
        s23 = -np.sqrt(beta_ly / beta_lx) * ((1.0 - u) * sin + alpha_lx * cos)
        s24 = ((alpha_ly * (1.0 - u) - alpha_lx * u) * sin + (u * (1.0 - u) + alpha_lx * alpha_ly) * cos) / np.sqrt(beta_lx * beta_ly)
    elif mode == 2:
        s11 = beta_lx
        s33 = beta_ly
        s12 = -alpha_lx
        s34 = -alpha_ly
        s22 = (u**2 + alpha_lx**2) / beta_lx
        s44 = ((1.0 - u) ** 2 + alpha_ly**2) / beta_ly
        s13 = np.sqrt(beta_lx * beta_ly) * cos
        s14 = -np.sqrt(beta_lx / beta_ly) * ((1.0 - u) * sin + alpha_ly * cos)
        s23 = np.sqrt(beta_ly / beta_lx) * (u * sin - alpha_lx * cos)
        s24 = ((alpha_lx * (1.0 - u) - alpha_ly * u) * sin + (u * (1.0 - u) + alpha_lx * alpha_ly) * cos) / np.sqrt(beta_lx * beta_ly)
    Sigma = eps * np.array([
            [s11, s12, s13, s14],
            [s12, s22, s23, s24],
            [s13, s23, s33, s34],
            [s14, s24, s34, s44],
        ])
    return Sigma