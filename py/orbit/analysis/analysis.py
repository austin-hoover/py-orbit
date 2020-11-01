import numpy as np
from numpy import linalg as la


def covmat2vec(S):
    return np.array([S[0,0], S[0,1], S[0,2], S[0,3],
                     S[1,1], S[1,2], S[1,3],
                     S[2,2], S[2,3],
                     S[3,3]])
                    

def coords(bunch, mm_mrad=True):
    """Return the transverse coordinate array from the bunch."""
    nparts = bunch.getSize()
    X = np.zeros((nparts, 4))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
    if mm_mrad:
        X *= 1000
    return X


class Stats:
    
    def __init__(self, file_path):
        """Open a new file to store the moments."""
        if file_path.endswith('/'):
            file_path = file_path[:-1]
        self.file = open(''.join([file_path, '/moments.dat']), 'a')

    def write(self, s, bunch, mm_mrad=True):
        """Add the current moments as a new line in the file."""
        X = coords(bunch, mm_mrad)
        moments = covmat2vec(np.cov(X.T))
        f = '{}' + 10*' {}' + '\n'
        self.file.write(f.format(
            s,
            moments[0], moments[1], moments[1], moments[3],
            moments[4], moments[5], moments[6],
            moments[7], moments[8],
            moments[9])
        )
        
    def close(self):
        self.file.close()
