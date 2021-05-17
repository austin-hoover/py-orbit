"""
This module contains functions related to the parameterization of coupled
motion developed by Lebedev and Bogacz.

Reference: https://iopscience.iop.org/article/10.1088/1748-0221/5/10/P10010
"""
import numpy as np
import numpy.linalg as la


# 4x4 nonsingular, skew-symmetric matrix. A matrix M is symplectic if 
# M^T U M = U.
U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

    
def normalize(eigvecs):
    """Normalize transfer matrix eigenvectors.
    
    eigvecs: ndarray, shape (4, 4)
        Each column is an eigenvector.
    """
    v1, _, v2, _ = eigvecs.T
    for i in (0, 2):
        v = eigvecs[:, i]
        val = la.multi_dot([np.conj(v), U, v]).imag
        if val > 0:
            eigvecs[:, i], eigvecs[:, i+1] = eigvecs[:, i+1], eigvecs[:, i]
        eigvecs[:, i:i+2] *= np.sqrt(2 / np.abs(val))
    return eigvecs
    
    
def construct_V(eigvecs):
    """Construct symplectic normalization matrix V from the eigenvectors.
    
    eigvecs: ndarray, shape (4, 4)
        Each column is an eigenvector.
    """
    v1, _, v2, _ = normalize(eigvecs).T
    V = np.zeros((4, 4))
    V[:, 0] = v1.real
    V[:, 1] = (1j * v1).real
    V[:, 2] = v2.real
    V[:, 3] = (1j * v2).real
    return V


def construct_Sigma(V, eps1, eps2):
    """Construct the matched covariance matrix using V.
    
    It will be matched to the transfer matrix defined by M = V P V^-1, where
    P rotates x-x' and y-y' phase spaces.
    
    V : ndarray, shape (4, 4)
        Symplectic normalization matrix.
    eps1, eps2, float
        Intrinsic beam emittances.
    """
    return la.multi_dot([V, np.diag([eps1, eps1, eps2, eps2]), V.T])
    
    
def matched_Sigma(M, eps1=1., eps2=1.):
    """Same as `construct_Sigma`, but computes V first."""
    eigvals, eigvecs = la.eig(M)
    V = construct_V(eigvecs)
    return construct_Sigma(V, eps1, eps2)
    

def matched_Sigma_onemode(M, eps, mode):
    """Construct the matched covariance matrix for distribution in which the
    one of the intrinsic emittances is zero."""
    eigvals, eigvecs = la.eig(M)
    V = construct_V(eigvecs)
    eps1 = eps if mode == 1 else 0
    eps2 = eps if mode == 2 else 0
    return construct_Sigma(V, eps1, eps2)
    
    
def extract_twiss(V):
    """"Extract 4D Twiss parameters from the definition of V."""
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
    a2x = (u*np.sin(nu2) - V[1, 2]*np.sqrt(b2x)) / np.cos(nu2)
    return a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y, u, nu1, nu2


def symplectic_diag(Sigma):
    """Perform symplectic diagonalization on the covariance matrix `Sigma`."""
    eigvals, eigvecs = la.eig(np.matmul(Sigma, U))
    Vinv = la.inv(construct_V(eigvecs))
    return la.multi_dot([Vinv, Sigma, Vinv.T])
    
    
def Vmat(ax, ay, bx, by, u, nu, mode=1):
    """Construct V from the Twiss parameters for the case when one of the
    intrinsic emittances is zero."""
    cos, sin = np.cos(nu), np.sin(nu)
    V = np.zeros((4, 4))
    if mode == 1:
        V[:2, :2] = [[np.sqrt(bx), 0],
                     [-ax/np.sqrt(bx), (1-u)/np.sqrt(bx)]]
        V[2:, :2] = [[np.sqrt(by)*cos, -np.sqrt(by)*sin],
                     [(u*sin-ay*cos)/np.sqrt(by), (u*cos + ay*sin)/np.sqrt(by)]]
    elif mode == 2:
        V[2:, 2:] = [[np.sqrt(by), 0],
                     [-ay/np.sqrt(by), (1-u)/np.sqrt(by)]]
        V[:2, 2:] = [[np.sqrt(bx)*cos, -np.sqrt(bx)*sin],
                     [(u*sin-ax*cos)/np.sqrt(bx), (u*cos + ax*sin)/np.sqrt(bx)]]
    return V
    
    
def Sigma(ax, ay, bx, by, u, nu, eps, mode=1):
    """Construct the covariance matrix matrix from the Twiss parameters for
    the case when one of the intrinsic emittances is zero."""
    cos, sin = np.cos(nu), np.sin(nu)
    if mode == 1:
        s11, s33 = bx, by
        s12, s34 = -ax, -ay
        s22 = ((1-u)**2 + ax**2) / bx
        s44 = (u**2 + ay**2) / by
        s13 = np.sqrt(bx*by) * cos
        s14 = np.sqrt(bx/by) * (u*sin - ay*cos)
        s23 = -np.sqrt(by/bx) * ((1-u)*sin + ax*cos)
        s24 = ((ay*(1-u) - ax*u)*sin + (u*(1-u) + ax*ay)*cos) / np.sqrt(bx*by)
    elif mode == 2:
        s11, s33 = bx, by
        s12, s34 = -ax, -ay
        s22 = (u**2 + ax**2) / bx
        s44 = ((1-u)**2 + ay**2) / by
        s13 = np.sqrt(bx*by) * cos
        s14 = -np.sqrt(bx/by) * ((1-u)*sin + ay*cos)
        s23 = np.sqrt(by/bx) * (u*sin - ax*cos)
        s24 = ((ax*(1-u) - ay*u)*sin + (u*(1-u) + ax*ay)*cos) / np.sqrt(bx*by)
    return eps * np.array([[s11, s12, s13, s14],
                           [s12, s22, s23, s24],
                           [s13, s23, s33, s34],
                           [s14, s24, s34, s44]])