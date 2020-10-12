"""Helper functions for envelope solver benchmarks."""

# Imports
#------------------------------------------------------------------------------
import numpy as np

from bunch import Bunch
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice


# General
#------------------------------------------------------------------------------
def rotation_matrix(phi):
    return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

def phase_space_rotation_matrix(phi_x, phi_y):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(phi_x)
    R[2:, 2:] = rotation_matrix(phi_y)
    return R

def norm_mat_2D(alpha, beta):
    return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)

def norm_mat_4D(alpha_x, beta_x, alpha_y, beta_y):
    V = np.zeros((4, 4))
    V[:2, :2] = norm_mat_2D(alpha_x, beta_x)
    V[2:, 2:] = norm_mat_2D(alpha_y, beta_y)
    return V

def get_perveance(energy, mass, density):
    gamma = 1 + (energy / mass) # Lorentz factor           
    beta = np.sqrt(1 - (1 / (gamma**2))) # v/c   
    r0 = 1.53469e-18 # classical proton radius [m]
    return (2 * r0 * density) / (beta**2 * gamma**3)

def create_lattice(madx_filename, seq):
    lattice = teapot.TEAPOT_Lattice()
    lattice.readMADX(madx_filename, seq)
    return lattice

def lattice_twiss(lattice, mass, energy):
    """Get lattice Twiss parameters at s=0."""
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    arrmuX, arrPosAlphaX, arrPosBetaX = matrix_lattice.getRingTwissDataX()
    arrmuY, arrPosAlphaY, arrPosBetaY = matrix_lattice.getRingTwissDataY()
    alpha_x, alpha_y = arrPosAlphaX[0][1], arrPosAlphaY[0][1]
    beta_x, beta_y = arrPosBetaX[0][1], arrPosBetaY[0][1]
    return alpha_x, beta_x, alpha_y, beta_y

def split_nodes(lattice, max_length):
    for node in lattice.getNodes():
        node_length = node.getLength()
        if node_length > max_length:
            node.setnParts(int(node_length / max_length))

def initialize_bunch(mass, energy):
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    return bunch, params_dict


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


# 2D rotating distribution 
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

class RotDist2D:
    
    def __init__(self, twiss_x, twiss_y, nu=np.pi/2, rot_dir='ccw'):
        
        (alpha_x, beta_x, eps_x), (alpha_y, beta_y, eps_y)= twiss_x, twiss_y
        R = phase_space_rotation_matrix(nu - np.pi/2, 0.0)
        A = np.sqrt(4 * np.diag([eps_x, eps_x, eps_y, eps_y]))
        V = norm_mat_4D(alpha_x, beta_x, alpha_y, beta_y)
        # Get envelope parameters
        if rot_dir == 'ccw':
            P = np.array([[1, 0], [0, 1], [0, -1], [+1, 0]])
        elif rot_dir == 'cw':
            P = np.array([[1, 0], [0, 1], [0, +1], [-1, 0]])
        self.params = np.linalg.multi_dot([V, A, R, P]).flatten()
        
    def get_coords(self):
        rho = np.sqrt(np.random.random())
        psi = 2 * np.pi * np.random.random()
        sin, cos = np.sin(psi), np.cos(psi)
        a, b, ap, bp, e, f, ep, fp = self.params
        x = rho * (a*sin + b*cos)
        y = rho * (e*sin + f*cos)
        xp = rho * (ap*sin + bp*cos)
        yp = rho * (ep*sin + fp*cos)
        return x, xp, y, yp