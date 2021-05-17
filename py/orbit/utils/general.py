"""
This module contains functions that should be of general use.
"""


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