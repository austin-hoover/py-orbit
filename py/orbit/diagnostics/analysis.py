import numpy as np
from numpy import linalg as la


def bunch_coord_array(bunch, mm_mrad=False, transverse_only=False):
    """Return bunch coordinate array."""
    n_parts = bunch.getSize()
    X = np.zeros((n_parts, 6))
    for i in range(n_parts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i),
                bunch.z(i), bunch.dE(i)]
    if mm_mrad:
        X[:, :4] *= 1000.
    if transverse_only:
        X = X[:, :4]
    return X
    
    
def to_vec(Sigma):
    """Return array of 10 unique elements of covariance matrix."""
    return Sigma[np.triu_indices(4)]

    
def rms_ellipse_dims(Sigma, x1='x', x2='y'):
    """Return (angle, c1, c2) of rms ellipse in x1-x2 plane, where angle is the
    clockwise tilt angle and c1/c2 are the semi-axes.
    """
    str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
    i, j = str_to_int[x1], str_to_int[x2]
    sii, sjj, sij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
    angle = -0.5 * np.arctan2(2*sij, sii-sjj)
    sin, cos = np.sin(angle), np.cos(angle)
    sin2, cos2 = sin**2, cos**2
    c1 = np.sqrt(abs(sii*cos2 + sjj*sin2 - 2*sij*sin*cos))
    c2 = np.sqrt(abs(sii*sin2 + sjj*cos2 + 2*sij*sin*cos))
    return angle, c1, c2
    
    
def intrinsic_emittances(Sigma):
    """Return intrinsic emittances from covariance matrix."""
    U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    trSU2 = np.trace(la.matrix_power(np.matmul(Sigma, U), 2))
    detS = la.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-trSU2 + np.sqrt(trSU2**2 - 16 * detS))
    eps_2 = 0.5 * np.sqrt(-trSU2 - np.sqrt(trSU2**2 - 16 * detS))
    return eps_1, eps_2
    
    
def apparent_emittances(Sigma):
    """Return apparent emittances from covariance matrix."""
    eps_x = np.sqrt(la.det(Sigma[:2, :2]))
    eps_y = np.sqrt(la.det(Sigma[2:, 2:]))
    return eps_x, eps_y
    
    
def twiss2D(Sigma):
    """Return 2D Twiss parameters from covariance matrix."""
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma[0, 0] / eps_x
    beta_y = Sigma[2, 2] / eps_y
    alpha_x = -Sigma[0, 1] / eps_x
    alpha_y = -Sigma[2, 3] / eps_y
    return [alpha_x, alpha_y, beta_x, beta_y]
        
    
class BunchStats:
    """Container for beam statistics."""
    def __init__(self, X):
        self.Sigma = np.cov(X.T)
        self.moments = to_vec(self.Sigma)
        self.eps_1, self.eps_2 = intrinsic_emittances(self.Sigma)
        self.eps_x, self.eps_y = apparent_emittances(self.Sigma)
        self.alpha_x, self.alpha_y, self.beta_x, self.beta_y = twiss2D(self.Sigma)
        
        
class DanilovEnvelopeBunch:
    """Stores the Danilov envelope parameters and test particle coordinates.
    
    Attributes
    -----------
    env_params : ndarray, shape (8,)
        The transverse beam envelope is defined by these eight parameters. They
        are stored using the first two bunch particles.
    coords : ndarray, shape (n_parts, 4)
        Each of the remaining bunch particles (third, fourth, etc.) respond
        to the space charge field defined by the envelope, but do not affect
        each other or the envelope itself. 
    """
    def __init__(self, X):
        (a, ap, e, ep), (b, bp, f, fp) = X[:2]
        self.env_params = np.array([a, b, ap, bp, e, f, ep, fp])
        self.coords = None
        if X.shape > 2:
            self.coords = X[2:]