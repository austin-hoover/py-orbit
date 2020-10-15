# Imports
#------------------------------------------------------------------------------

# Python
import numpy as np
from tqdm import trange

# PyORBIT
from bunch import Bunch
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.bunch_generators import TwissContainer, TwissAnalysis
from orbit.bunch_generators import WaterBagDist2D, GaussDist2D, KVDist2D


#------------------------------------------------------------------------------

def get_perveance(energy, mass, density):
    gamma = 1 + (energy / mass) # Lorentz factor           
    beta = np.sqrt(1 - (1 / (gamma**2))) # v/c   
    r0 = 1.53469e-18 # classical proton radius [m]
    return (2 * r0 * density) / (beta**2 * gamma**3)


def mat2vec(S):
    """Return vector of independent elements in 4x4 symmetric matrix S."""   
    return np.array([S[0,0], S[0,1], S[0,2], S[0,3], S[1,1], 
                     S[1,2], S[1,3], S[2,2], S[2,3], S[3,3]])


def vec2mat(v):
    """Return symmetric matrix from vector."""
    S11, S12, S13, S14, S22, S23, S24, S33, S34, S44 = v
    return np.array([[S11, S12, S13, S14],
                     [S12, S22, S23, S24],
                     [S13, S23, S33, S34],
                     [S14, S24, S34, S44]])
         
    
def set_fringe(lattice, fringe=False):
    """Turn fringe fields on/off."""
    for node in lattice.getNodes():
        node.setUsageFringeFieldIN(fringe)
        node.setUsageFringeFieldOUT(fringe)
    
    
def lattice_from_file(file, seq, fringe=False):
    """Create lattice from madx file and turn on(off) fringe fields."""
    lattice = teapot.TEAPOT_Lattice()
    lattice.readMADX(file, seq)
    set_fringe(lattice, fringe)
    return lattice


def split_nodes(lattice, max_length):
    for node in lattice.getNodes():
        node_length = node.getLength()
        if node_length > max_length:
            node.setnParts(int(node_length / max_length))
    
    
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
    
    
def initialize_envelope(env_params, mass, energy):
    """Return Bunch object for envelope.
    
    Inputs
    ------
    env_params : array-like
        The envelope parameters (a, b, ap, bp, e, f, ep, fp)
    mass : float
        The particle mass [GeV].
    energy : float
        The kinetic energy per particle [GeV]
        
    Returns
    -------
    bunch : Bunch object
    params_dict : dict
        The parameters dictionary for the bunch.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    a, b, ap, bp, e, f, ep, fp = env_params
    bunch.addParticle(a, ap, e, ep, 0, 0) 
    bunch.addParticle(b, bp, f, fp, 0, 0)
    return bunch, params_dict
        
    
def coasting_beaam(
    nparts,
    twiss_params, 
    lattice_length, 
    mass, 
    energy,          
    kind, macrosize=1
):  
    # Create bunch
    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(macrosize)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    # Create distribution generator
    distributions = {
        'kv':KVDist2D,
        'gaussian':GaussDist2D,
        'rotating':SCDist2D,
        'waterbag':WaterBagDist2D
    }
    ax, bx, ex, ay, by, ey = twiss_params
    dist = distributions[kind](
        TwissContainer(ax, bx, ex), 
        TwissContainer(ay, by, ey)
    )
    # Add particles to bunch
    for i in range(int(nparts)):
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
            
    
def get_env_params(bunch):
    """Extract envelope parameters from bunch."""
    a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
    b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
    return np.array([a, b, ap, bp, e, f, ep, fp])
        
        
def track_env(bunch, params_dict, lattice, nturns, output_file=None):
    """Track envelope through lattice."""
    env_params = np.zeros((nturns + 1, 8))
    for i in trange(nturns + 1):
        env_params[i] = get_env_params(bunch)
        lattice.trackBunch(bunch, params_dict)
    env_params *= 1000 # mm*mrad
    if output_file is not None:
        np.savetxt(output_file, env_params, fmt='%1.15f')
    return env_params
        
    
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
        Columns are: s, nux, nuy, alpha_x, alpha_x, beta_x, beta_y. The number
        of rows is dependent on the length of the lattice.
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
