"""
This module contains functions for use in PyORBIT scripts.
"""

# 3rd party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange

# PyORBIT modules
from bunch import Bunch
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.bunch_generators import (
    TwissContainer,
    WaterBagDist2D,
    GaussDist2D,
    KVDist2D
)
from orbit.utils.consts import classical_proton_radius
from orbit_utils import Matrix


#------------------------------------------------------------------------------
         
def tprint(string, indent=0):
    print indent*' ' + str(string)
    
         
def get_perveance(energy, mass, density):
    gamma = 1 + (energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / (gamma**2))) # v/c
    return (2 * classical_proton_radius * density) / (beta**2 * gamma**3)
    
    
def lattice_from_file(file, seq='', fringe=False):
    """Create lattice from madx file and turn on(off) fringe fields."""
    lattice = teapot.TEAPOT_Lattice()
    lattice.readMADX(file, seq)
    lattice.set_fringe(fringe)
    return lattice
    

def tilt_quads(lattice, angle):
    """Tilt all focusing{defocusing} quads by +angle{-angle}."""
    angle = np.radians(angle)
    qf_nodes = lattice.get_nodes_containing('qf')
    qd_nodes = lattice.get_nodes_containing('qd')
    for qf_node, qd_node in zip(qf_nodes, qd_nodes):
        qf_node.setTiltAngle(+angle)
        qf_node.setTiltAngle(-angle)
    

def fodo_lattice(mux, muy, L, fill_fac, angle=0, fringe=False):
    """Create (O-F-O-O-D-O) quadrupole lattice.
    
    Parameters
    ----------
    mux{y}: float
        The x{y} lattice phase advance [deg]. These are the phase advances
        when the lattice is uncoupled (`angle` == 0).
    L : float
        The length of the lattice.
    fill_fac : float
        The fraction of the lattice occupied by quadrupoles.
    angle : float
        The skew or tilt angle of the quads [deg]. The focusing
        quad is rotated clockwise by angle, and the defocusing quad is
        rotated counterclockwise by angle.
    fringe : bool
        Whether to include nonlinear fringe fields in the lattice.
    """
    angle = np.radians(angle)

    def fodo(k1, k2):
        """Create FODO lattice. k1 and k2 are the focusing strengths of the
        focusing (1st) and defocusing (2nd) quads, respectively.
        """
        lattice = TEAPOT_Lattice()
        drifts = [
            teapot.DriftTEAPOT('drift1')]
        drift1 = teapot.DriftTEAPOT('drift1')
        drift2 = teapot.DriftTEAPOT('drift2')
        drift3 = teapot.DriftTEAPOT('drift3')
        qf = teapot.QuadTEAPOT('qf')
        qd = teapot.QuadTEAPOT('qd')
        qf.addParam('kq', +k1)
        qd.addParam('kq', -k2)
        drift1.setLength(L * (1 - fill_fac) / 4)
        drift2.setLength(L * (1 - fill_fac) / 2)
        drift3.setLength(L * (1 - fill_fac) / 4)
        qf.setLength(L * fill_fac / 2)
        qd.setLength(L * fill_fac / 2)
        qf.setTiltAngle(+angle)
        qd.setTiltAngle(-angle)
        lattice.addNode(drift1)
        lattice.addNode(qf)
        lattice.addNode(drift2)
        lattice.addNode(qd)
        lattice.addNode(drift3)
        lattice.set_fringe(fringe)
        lattice.initialize()
        return lattice

    def cost(kvals, correct_tunes, mass=0.93827231, energy=1):
        lattice = fodo(*kvals)
        M = transfer_matrix(lattice, mass, energy)
        return correct_phase_adv - np.degrees(eigtunes(M)[[0, 2]])

    correct_phase_adv = np.array([mux, muy])
    k0 = np.array([0.5, 0.5]) # ~ 80 deg phase advance
    result = opt.least_squares(cost, k0, args=(correct_phase_adv,))
    k1, k2 = result.x
    return fodo(k1, k2)
    
    
def fofo_lattice(ks1, ks2, L, fill_fac, fringe=False):
    """Create O-F-O-O-F-O solenoid lattice.
    
    Parameters
    ----------
    ks1{2}: float
        The field strength of the 1st{2nd} solenoid.
    L : float
        The length of the lattice.
    fill_fac : float
        The fraction of the lattice occupied by quadrupoles.
    fringe : bool
        Whether to include nonlinear fringe fields in the lattice.
    """
    lattice = TEAPOT_Lattice()
    drift1 = teapot.DriftTEAPOT('drift1')
    drift2 = teapot.DriftTEAPOT('drift2')
    drift3 = teapot.DriftTEAPOT('drift3')
    sol1 = teapot.SolenoidTEAPOT('sol1')
    sol2 = teapot.SolenoidTEAPOT('sol2')
    sol1.addParam('B', ks1)
    sol2.addParam('B', ks2)
    drift1.setLength(L * (1 - fill_fac) / 4)
    drift2.setLength(L * (1 - fill_fac) / 2)
    drift3.setLength(L * (1 - fill_fac) / 4)
    sol1.setLength(L * fill_fac / 2)
    sol2.setLength(L * fill_fac / 2)
    lattice.addNode(drift1)
    lattice.addNode(sol1)
    lattice.addNode(drift2)
    lattice.addNode(sol2)
    lattice.addNode(drift3)
    lattice.set_fringe(fringe)
    lattice.initialize()
    return lattice
    
    
def transfer_matrix(lattice, mass, energy):
    """Get linear transfer matrix as NumPy array."""
    matrixGenerator = MatrixGenerator()
    bunch, params_dict = initialize_bunch(mass, energy)
    matrixGenerator.initBunch(bunch)
    lattice.trackBunch(bunch)
    transfer_mat = Matrix(6, 6)
    matrixGenerator.calculateMatrix(bunch, transfer_mat)
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = transfer_mat.get(i, j)
    return M
    

def params_from_transfer_matrix(M):
    """Return dict of lattice parameters from the transfer matrix.
    
    The method is taken from py/orbit/matrix_lattice/MATRIX_Lattice.py
    """
    keys = ['frac_tune_x', 'frac_tune_y', 'alpha_x', 'alpha_y', 'beta_x',
            'beta_y', 'gamma_x', 'gamma_y']
    lattice_params = {key: None for key in keys}
    
    cos_phi_x = (M[0, 0] + M[1, 1]) / 2
    cos_phi_y = (M[2, 2] + M[3, 3]) / 2
    if abs(cos_phi_x) >= 1 or abs(cos_phi_y) >= 1 :
        return lattice_params
    sign_x = sign_y = +1
    if abs(M[0, 1]) != 0:
        sign_x = M[0, 1] / abs(M[0, 1])
    if abs(M[2, 3]) != 0:
        sign_y = M[2, 3] / abs(M[2, 3])
    sin_phi_x = sign_x * np.sqrt(1 - cos_phi_x**2)
    sin_phi_y = sign_y * np.sqrt(1 - cos_phi_y**2)
    
    nux = sign_x * np.arccos(cos_phi_x) / (2 * np.pi)
    nuy = sign_y * np.arccos(cos_phi_y) / (2 * np.pi)
    beta_x = M[0, 1] / sin_phi_x
    beta_y = M[2, 3] / sin_phi_y
    alpha_x = (M[0, 0] - M[1, 1]) / (2 * sin_phi_x)
    alpha_y = (M[2, 2] - M[3, 3]) / (2 * sin_phi_y)
    gamma_x = -M[1, 0] / sin_phi_x
    gamma_y = -M[3, 2] / sin_phi_y
    
    lattice_params['frac_tune_x'] = nux
    lattice_params['frac_tune_y'] = nuy
    lattice_params['beta_x'] = beta_x
    lattice_params['beta_y'] = beta_y
    lattice_params['alpha_x'] = alpha_x
    lattice_params['alpha_y'] = alpha_y
    lattice_params['gamma_x'] = gamma_x
    lattice_params['gamma_y'] = gamma_y
    return lattice_params
    
    
def is_stable(M):
    """Determine stability of transfer matrix M."""
    for eigval in la.eigvals(M):
        if abs(la.norm(eigval) - 1) > 1e-5:
            return False
    return True
    
    
def eigtunes(M):
    return np.arccos(la.eigvals(M).real)
    
    
def twiss_at_injection(lattice, mass, energy):
    """Get the Twiss parameters at s=0 in lattice."""
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    _, arrPosAlphaX, arrPosBetaX = matrix_lattice.getRingTwissDataX()
    _, arrPosAlphaY, arrPosBetaY = matrix_lattice.getRingTwissDataY()
    alpha_x, alpha_y = arrPosAlphaX[0][1], arrPosAlphaY[0][1]
    beta_x, beta_y = arrPosBetaX[0][1], arrPosBetaY[0][1]
    return alpha_x, alpha_y, beta_x, beta_y
    
    
def twiss_throughout(lattice, bunch):
    """Get the Twiss parameters throughout lattice.
    
    Returns: NumPy array
        Columns are: s, nux, nuy, alpha_x, alpha_x, beta_x, beta_y.
        The number of rows is dependent on the length of the lattice.
    """
    # Extract Twiss parameters from one turn transfer matrix
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    twiss_x = matrix_lattice.getRingTwissDataX()
    twiss_y = matrix_lattice.getRingTwissDataY()
    # Unpack and convert to numpy arrays
    (nux, alpha_x, beta_x), (nuy, alpha_y, beta_y) = twiss_x, twiss_y
    nux, alpha_x, beta_x = np.array(nux), np.array(alpha_x), np.array(beta_x)
    nuy, alpha_y, beta_y = np.array(nuy), np.array(alpha_y), np.array(beta_y)
    # Merge into one array
    s = nux[:,0]
    nux, alpha_x, beta_x = nux[:,1], alpha_x[:,1], beta_x[:,1]
    nuy, alpha_y, beta_y = nuy[:,1], alpha_y[:,1], beta_y[:,1]
    return np.vstack([s, nux, nuy, alpha_x, alpha_y, beta_x, beta_y]).T

    
def add_node_at_start(lattice, node):
    """Add node at start of first node in lattice."""
    firstnode = lattice.getNodes()[0]
    firstnode.addChildNode(node, firstnode.ENTRANCE)


def add_node_at_end(lattice, node):
    """Add node at end of last node in lattice."""
    lastnode = lattice.getNodes()[-1]
    lastnode.addChildNode(node, lastnode.EXIT)


def add_node_throughout(lattice, new_node, position):
    """Add new_node as child of every node in lattice.
    
    position : str
        Options are {'start', 'mid', 'end'}.
    """
    loc = {
        'start': AccNode.ENTRANCE, 
        'mid': AccNode.BODY, 
        'end': AccNode.EXIT
    }
    for node in lattice.getNodes():
        node.addChildNode(new_node, loc[position], 0, AccNode.BEFORE)


def initialize_bunch(mass, energy):
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    return bunch, params_dict
        
    
def coasting_beaam(
    nparts,
    twiss_params, 
    lattice_length, 
    mass, 
    energy,          
    kind, macrosize=1
):  
    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(macrosize)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    distributions = {
        'kv':KVDist2D,
        'gaussian':GaussDist2D,
        'waterbag':WaterBagDist2D
    }
    ax, bx, ex, ay, by, ey = twiss_params
    dist = distributions[kind](
        TwissContainer(ax, bx, ex), 
        TwissContainer(ay, by, ey)
    )
    for i in range(nparts):
        x, xp, y, yp = dist.getCoordinates()
        z = lattice_length * np.random.random()
        bunch.addParticle(x, xp, y, yp, z, 0.0)
    return bunch, params_dict
                                                                    
    
def get_coords(bunch, mm_mrad=False):
    """Extract NumPy array of shape (nparts, 6) from bunch."""
    nparts = bunch.getSize()
    X = np.zeros((nparts, 6))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), 
                bunch.yp(i), bunch.z(i), bunch.dE(i)]
    if mm_mrad:
        X *= 1000.
    return X
    
    
def dist_to_bunch(X, bunch, length):
    for (x, xp, y, yp) in X:
        z = length * np.random.random()
        bunch.addParticle(x, xp, y, yp, z, 0.)
    return bunch
    
    
def track_bunch(bunch, params_dict, lattice, nturns, output_dir, dump_every=0):
    """Track bunch through lattice."""
    
    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]
      
    coords = []
    for i in trange(nturns + 1):
        if dump_every > 0 and i % dump_every == 0:
            filename = ''.join([output_dir, '/coords_{}.dat'.format(i)])
            X = get_coords(bunch, mm_mrad=True)
            np.savetxt(filename, X)
            coords.append(X)  
        lattice.trackBunch(bunch, params_dict)
              
    return np.array(coords)
