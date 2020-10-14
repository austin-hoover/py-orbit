"""Functions for envelope solver benchmarks."""

# Imports
#------------------------------------------------------------------------------
import numpy as np

from bunch import Bunch
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice


# Bunch analysis 
#------------------------------------------------------------------------------
def coords(bunch):
    """Extract transverse coordinate matrix from bunch."""
    nparts = bunch.getSize()
    X = np.zeros((nparts, 4))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.px(i), bunch.y(i), bunch.py(i)]
    return X


def cov_mat(bunch):
    """Compute transverse covariance matrix from bunch."""
    X = coords(bunch)
    return np.cov(X.T)


def radii(bunch):
    """Compute radii and slopes of KV envelope from bunch."""
    sigma = cov_mat(bunch)
    rx = 2 * np.sqrt(sigma[0, 0])
    ry = 2 * np.sqrt(sigma[2, 2])
    rxp = sigma[0, 1] / np.sqrt(sigma[0, 0])
    ryp = sigma[2, 3] / np.sqrt(sigma[2, 2])
    return np.array([rx, rxp, ry, ryp])


def radii_from_twiss(alpha, beta, eps):
    """Return radius and slope of KV envelope from Twiss parameters."""
    r = np.sqrt(4 * eps * beta)
    rp = -alpha * np.sqrt(4 * eps / beta)
    return r, rp


# Danilov distribution
#------------------------------------------------------------------------------
def get_env_params(bunch):
    """Extract rotating envelope parameters from bunch."""
    a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
    b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
    return [a, b, ap, bp, e, f, ep, fp]


def env_cov_mat(bunch):
    """Construct transverse covariance matrix from envelope parameters."""
    a, b, ap, bp, e, f, ep, fp = get_env_params(bunch)
    M = np.array([[a, b, 0, 0], [ap, bp, 0, 0], [e, f, 0, 0], [ep, fp, 0, 0]])
    return 0.25 * np.matmul(M, M.T)
