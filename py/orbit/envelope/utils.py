import numpy as np

from bunch import Bunch
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils import consts


def rotation_matrix(angle):
    """2 x 2 clockwise rotation matrix (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4x4(angle):
    """4 x 4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_advance_matrix(*phase_advances):
    """Phase advance matrix (clockwise rotation in each phase plane).

    Parameters
    ---------
    mu1, mu2, ..., mun : float
        The phase advance in each plane.

    Returns
    -------
    ndarray, shape (2n, 2n)
        Matrix which rotates x-x', y-y', z-z', etc. by the phase advances.
    """
    n = len(phase_advances)
    M = np.zeros((2 * n, 2 * n))
    for i, phase_advance in enumerate(phase_advances):
        i = i * 2
        M[i : i + 2, i : i + 2] = rotation_matrix(phase_advance)
    return M


def get_transfer_matrix(lattice=None, mass=None, kin_energy=None):
    """Return linear 6x6 transfer matrix from periodic lattice as ndarray.

    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].

    Returns
    -------
    M : ndarray, shape (6, 6)
        Transverse transfer matrix.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            M[i, j] = matrix_lattice.oneTurnMatrix.get(i, j)
    return M


def get_perveance(mass=None, kin_energy=None, line_density=None):
    """ "Compute dimensionless beam perveance.

    Parameters
    ----------
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    line_density : float
        Number density in longitudinal direction [m^-1].

    Returns
    -------
    float
        Dimensionless space charge perveance
    """
    classical_proton_radius = 1.53469e-18  # [m]
    gamma = 1.0 + (kin_energy / mass)  # Lorentz factor
    beta = np.sqrt(1.0 - (1.0 / gamma) ** 2)  # velocity/speed_of_light
    return (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)


def emittance_2x2(Sigma):
    """RMS emittance from u-u' covariance matrix.

    Parameters
    ----------
    Sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    float
        The RMS emittance (sqrt(<uu><u'u'> - <uu'>^2)).
    """
    return np.sqrt(np.linalg.det(Sigma))


def twiss_2x2(Sigma):
    """RMS Twiss parameters from 2 x 2 covariance matrix.

    Parameters
    ----------
    Sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    alpha : float
        The alpha parameter (-<uu'> / sqrt(<uu><u'u'> - <uu'>^2)).
    beta : float
        The beta parameter (<uu> / sqrt(<uu><u'u'> - <uu'>^2)).
    """
    eps = emittance_2x2(Sigma)
    beta = Sigma[0, 0] / eps
    alpha = -Sigma[0, 1] / eps
    return alpha, beta