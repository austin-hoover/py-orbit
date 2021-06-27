"""
This module contains functions that should be of general use.
"""
import numpy as np


def tprint(string, indent=4):
    """Print with indent."""    
    print indent * ' ' + str(string)
    
    
def step_func(x):
    "Heaviside step function."
    return 1 if x >= 0 else 0
    
    
def apply(M, X):
    """Apply matrix M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def normalize(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x/la.norm(x), 1, X)


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


# The following three functions are from Tony Yu's blog post: https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR
    
def stack_ragged(array_list, axis=0):
    """Stacks list of arrays along first axis.
    
    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(array)[axis] for array in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx
    

def save_stacked_array(filename, array_list, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)
    
    
def load_stacked_arrays(filename, axis=0):
    """"Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file['stacked_index']
    stacked = npz_file['stacked_array']
    return np.split(stacked, idx, axis=axis)
