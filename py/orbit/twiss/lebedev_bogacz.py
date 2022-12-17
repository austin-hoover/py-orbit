"""Lebedev-Bogacz parameterization of coupled motion.

Under developement. Some of these methods have not been tested in a while.

References
----------
[1] https://iopscience.iop.org/article/10.1088/1748-0221/5/10/P10010
"""
import numpy as np


def normalize(eigvecs):
    """Normalize transfer matrix eigenvectors.

    eigvecs: ndarray, shape (2n, 2n)
        Each column is an eigenvector.
    """
    n = eigvecs.shape[0]
    for i in range(0, n, 2):
        v = eigvecs[:, i]
        # Find out if we need to swap.
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0:
            eigvecs[:, i], eigvecs[:, i + 1] = eigvecs[:, i + 1], eigvecs[:, i]
        eigvecs[:, i : i + 2] *= np.sqrt(2 / np.abs(val))
    return eigvecs


def norm_mat_from_eigvecs(eigvecs):
    """Construct symplectic normalization matrix.

    eigvecs: ndarray, shape (2n, 2n)
        Each column is an eigenvector. We assume they are not normalized.
    """
    eigvecs = normalize(eigvecs.copy())
    V = np.zeros(eigvecs.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigvecs[:, i].real
        V[:, i + 1] = (1j * eigvecs[:, i]).real
    return V


def norm_mat_from_twiss_one_mode(alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu, mode=1):
    cos, sin = np.cos(nu), np.sin(nu)
    V = np.zeros((4, 4))
    if mode == 1:
        V[:2, :2] = [
            [np.sqrt(beta_lx), 0],
            [-alpha_lx / np.sqrt(beta_lx), (1 - u) / np.sqrt(beta_lx)],
        ]
        V[2:, :2] = [
            [np.sqrt(beta_ly) * cos, -np.sqrt(beta_ly) * sin],
            [
                (u * sin - alpha_ly * cos) / np.sqrt(beta_ly),
                (u * cos + alpha_ly * sin) / np.sqrt(beta_ly),
            ],
        ]
    elif mode == 2:
        V[2:, 2:] = [
            [np.sqrt(beta_ly), 0],
            [-alpha_ly / np.sqrt(beta_ly), (1 - u) / np.sqrt(beta_ly)],
        ]
        V[:2, 2:] = [
            [np.sqrt(beta_lx) * cos, -np.sqrt(beta_lx) * sin],
            [
                (u * sin - alpha_lx * cos) / np.sqrt(beta_lx),
                (u * cos + alpha_lx * sin) / np.sqrt(beta_lx),
            ],
        ]
    return V


def twiss_from_norm_mat(V):
    beta_1x = V[0, 0] ** 2
    beta_2y = V[2, 2] ** 2
    alpha_1x = -np.sqrt(beta_1x) * V[1, 0]
    alpha_2y = -np.sqrt(beta_2y) * V[3, 2]
    u = 1.0 - V[0, 0] * V[1, 1]
    nu1 = np.arctan2(-V[2, 1], V[2, 0])
    nu2 = np.arctan2(-V[0, 3], V[0, 2])
    beta_1y = (V[2, 0] / np.cos(nu1)) ** 2
    beta_2x = (V[0, 2] / np.cos(nu2)) ** 2
    alpha_1y = (u * np.sin(nu1) - V[3, 0] * np.sqrt(beta_1y)) / np.cos(nu1)
    alpha_2x = (u * np.sin(nu2) - V[1, 2] * np.sqrt(beta_2x)) / np.cos(nu2)
    return (
        alpha_1x,
        beta_1x,
        alpha_1y,
        beta_1y,
        alpha_2x,
        beta_2x,
        alpha_2y,
        beta_2y,
        u,
        nu1,
        nu2,
    )


def analyze_transfer_matrix(M):
    """Return parameter dict from 4 x 4 transfer matrix."""
    eigvals, eigvecs = np.linalg.eig(M)
    V = norm_mat_from_eigvecs(eigvecs)
    params = dict()
    (
        params["alpha_1x"],
        params["beta_1x"],
        params["alpha_1y"],
        params["beta_1y"],
        params["alpha_2x"],
        params["beta_2x"],
        params["alpha_2y"],
        params["beta_2y"],
        params["u"],
        params["nu1"],
        params["nu2"],
    ) = twiss_from_norm_mat(V)
    params["V"] = V


def matched_cov(M, *intrinsic_emittances):
    eigvals, eigvecs = np.linalg.eig(M)
    V = norm_mat_from_eigvecs(eigvecs)
    Sigma_n = np.diag(np.repeat(intrinsic_emittances, 2))
    return np.linalg.multi_dot([V, Sigma_n, V.T])


def cov_from_twiss_one_mode(alpha_lx, alpha_ly, beta_lx, beta_ly, u, nu, eps, mode=1):
    cos, sin = np.cos(nu), np.sin(nu)
    if mode == 1:
        s11, s33 = beta_lx, beta_ly
        s12, s34 = -alpha_lx, -alpha_ly
        s22 = ((1 - u) ** 2 + alpha_lx**2) / beta_lx
        s44 = (u**2 + alpha_ly**2) / beta_ly
        s13 = np.sqrt(beta_lx * beta_ly) * cos
        s14 = np.sqrt(beta_lx / beta_ly) * (u * sin - alpha_ly * cos)
        s23 = -np.sqrt(beta_ly / beta_lx) * ((1 - u) * sin + alpha_lx * cos)
        s24 = (
            (alpha_ly * (1 - u) - alpha_lx * u) * sin
            + (u * (1 - u) + alpha_lx * alpha_ly) * cos
        ) / np.sqrt(beta_lx * beta_ly)
    elif mode == 2:
        s11, s33 = beta_lx, beta_ly
        s12, s34 = -alpha_lx, -alpha_ly
        s22 = (u**2 + alpha_lx**2) / beta_lx
        s44 = ((1 - u) ** 2 + alpha_ly**2) / beta_ly
        s13 = np.sqrt(beta_lx * beta_ly) * cos
        s14 = -np.sqrt(beta_lx / beta_ly) * ((1 - u) * sin + alpha_ly * cos)
        s23 = np.sqrt(beta_ly / beta_lx) * (u * sin - alpha_lx * cos)
        s24 = (
            (alpha_lx * (1 - u) - alpha_ly * u) * sin
            + (u * (1 - u) + alpha_lx * alpha_ly) * cos
        ) / np.sqrt(beta_lx * beta_ly)
    return eps * np.array(
        [
            [s11, s12, s13, s14],
            [s12, s22, s23, s24],
            [s13, s23, s33, s34],
            [s14, s24, s34, s44],
        ]
    )


def symplectic_diag(Sigma):
    """Symplectic diagonalization of covariance matrix `Sigma`."""
    U = unit_symplectic_matrix(4)
    eigvals, eigvecs = np.linalg.eig(np.matmul(Sigma, U))
    Vinv = np.linalg.inv(norm_mat_from_eigvecs(eigvecs))
    return np.linalg.multi_dot([Vinv, Sigma, Vinv.T])
