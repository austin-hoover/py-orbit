import os
import sys
import numpy as np


def delete_files_not_folders(path):
    for root, folders, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


def tprint(string, indent=4):
    """Print with indent."""    
    print indent * ' ' + str(string)
    
    
def apply(M, X):
    """Apply matrix M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def normalize(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, X)


def symmetrize(M):
    """Return symmetrized version of square upper/lower triangular matrix."""
    return M + M.T - np.diag(M.diagonal())
    
    
def rand_rows(X, n):
    """Return n random elements of X."""
    Xsamp = np.copy(X)
    if n < X.shape[0]:
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp
    
    
def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4D(angle):
    """4x4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])