"""
This module contains functions related to the coupling parameterization of
Bogacz and Lebedev.
"""

import numpy as np
import numpy.linalg as la
from tqdm import trange

U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

def rotation_matrix(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])
    
def phase_adv_matrix(mu1, mu2):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(mu1)
    R[2:, 2:] = rotation_matrix(mu2)
    return R
    
def normalize(eigvecs):
    """Normalize of transfer matrix eigenvectors."""
    v = eigvecs[:, 0]
    val = la.multi_dot([np.conj(v), U, v]).imag
    if val > 0:
        eigvecs[:, 0], eigvecs[:, 1] = eigvecs[:, 1], eigvecs[:, 0]
        eigvecs[:, 2], eigvecs[:, 3] = eigvecs[:, 3], eigvecs[:, 2]
    return eigvecs * np.sqrt(2 / np.abs(val))

def construct_V(M):
    """Construct normalization matrix from transfer matrix."""
    eigvals, eigvecs = la.eig(M)
    eigvecs = normalize(eigvecs)
    v1, _, v2, _ = eigvecs.T
    V = np.zeros((4, 4))
    V[:, 0] = v1.real
    V[:, 1] = (1j * v1).real
    V[:, 2] = v2.real
    V[:, 3] = (1j * v2).real
    return V

def construct_Sigma(V, e1, e2):
    return la.multi_dot([V, np.diag([e1, e1, e2, e2]), V.T])

def get_matched_Sigma(M, e1=1., e2=1.):
    return construct_Sigma(construct_V(M), e1, e2)
    
def extract_twiss(V):
    b1x = V[0, 0]**2
    b2y = V[2, 2]**2
    a1x = -np.sqrt(b1x) * V[1, 0]
    a2y = -np.sqrt(b2y) * V[3, 2]
    u = 1 - V[0, 0] * V[1, 1]
    nu1 = np.arctan2(-V[2, 1], V[2, 0])
    nu2 = np.arctan2(-V[0, 3], V[0, 2])
    b1y = (V[2, 0] / np.cos(nu1))**2
    b2x = (V[0, 2] / np.cos(nu2))**2
    a1y = (u*np.sin(nu1) - V[3, 0]*np.sqrt(b1y)) / np.cos(nu1)
    a1y = (u*np.sin(nu1) - V[3, 0]*np.sqrt(b1y)) / np.cos(nu1)
    a2x = (u*np.sin(nu2) - V[1, 2]*np.sqrt(b2x)) / np.cos(nu2)
    return a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y, u, nu1, nu2
