import numpy as np

from orbit_utils import Matrix


def orbit_matrix_to_numpy(matrix):
    """Return ndarray from two-dimensional orbit matrix."""
    array = np.zeros(matrix.size())
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.get(i, j)
    return array