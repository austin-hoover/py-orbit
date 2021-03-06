"""
This module contains functions for use in PyORBIT scripts.
"""
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange

from bunch import Bunch
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import KVDist2D
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.twiss.twiss import get_eigtunes, params_from_transfer_matrix
from orbit.utils.consts import classical_proton_radius, speed_of_light
from orbit_utils import Matrix

             
def lattice_from_file(file, seq='', fringe=False, kind='madx'):
    """Shortcut to create TEAPOT_Lattice from MAD or MADX file.
    
    Parameters
    ----------
    file : str
        '.lat' file output from MADX script.
    seq : str
        Key word for start of beam line definition in `file`. In MAD this is
        called 'line'.
    fringe : bool
        Whether to turn on fringe fields.
    kind : {'madx', 'mad'}
        File format.
    
    Returns
    -------
    TEAPOT_Lattice
    """
    lattice = teapot.TEAPOT_Lattice()
    if kind == 'madx':
        lattice.readMADX(file, seq)
    elif kind == 'mad':
        line = seq
        lattice.readMAD(file, line)
    lattice.set_fringe(fringe)
    return lattice
    
    
def get_perveance(mass, kin_energy, line_density):
    """"Compute dimensionless beam perveance.
    
    Parameters
    ----------
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    line_density : float
        Number density in longitudinal direction [m^-1].
    
    Returns
    -------
    float
        Dimensionless space charge perveance
    """
    gamma = 1 + (kin_energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma)**2) # velocity/speed_of_light
    return (2 * classical_proton_radius * line_density) / (beta**2 * gamma**3)
    
    
def get_intensity(perveance, mass, kin_energy, bunch_length):
    """Return intensity from perveance."""
    gamma = 1 + (kin_energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma)**2) # velocity/speed_of_light
    return beta**2 * gamma**3 * perveance / (2 * classical_proton_radius)
    
    
def get_Brho(mass, kin_energy):
    """Compute magnetic rigidity [T * m]/
    
    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].
    """
    pc = get_pc(mass, kin_energy)
    return 1e9 * (pc / speed_of_light)
    
    
def get_pc(mass, kin_energy):
    """Return momentum * speed_of_light [GeV].
    
    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].
    """
    return np.sqrt(kin_energy * (kin_energy + 2 * mass))


def fodo_lattice(mux, muy, L, fill_fac=0.5, angle=0, start='drift', fringe=False,
                 reverse=False):
    """Create a FODO lattice.
    
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
    start : str
        If 'drift', the lattice will be O-F-O-O-D-O. If 'quad' the lattice will
        be (F/2)-O-O-D-O-O-(F/2).
    reverse : bool
        If True, reverse the lattice elements. This places the defocusing quad
        first.
    
    Returns
    -------
    TEAPOT_Lattice
    """
    angle = np.radians(angle)

    def fodo(k1, k2):
        """Create FODO lattice. k1 and k2 are the focusing strengths of the
        focusing (1st) and defocusing (2nd) quads, respectively.
        """
        # Instantiate elements
        lattice = TEAPOT_Lattice()
        drift1 = teapot.DriftTEAPOT('drift1')
        drift2 = teapot.DriftTEAPOT('drift2')
        drift_half1 = teapot.DriftTEAPOT('drift_half1')
        drift_half2 = teapot.DriftTEAPOT('drift_half2')
        qf = teapot.QuadTEAPOT('qf')
        qd = teapot.QuadTEAPOT('qd')
        qf_half1 = teapot.QuadTEAPOT('qf_half1')
        qf_half2 = teapot.QuadTEAPOT('qf_half2')
        qd_half1 = teapot.QuadTEAPOT('qd_half1')
        qd_half2 = teapot.QuadTEAPOT('qd_half2')
        # Set lengths
        half_nodes = (drift_half1, drift_half2, qf_half1,
                      qf_half2, qd_half1, qd_half2)
        full_nodes = (drift1, drift2, qf, qd)
        for node in half_nodes:
            node.setLength(L * fill_fac / 4)
        for node in full_nodes:
            node.setLength(L * fill_fac / 2)
        # Set quad focusing strengths
        for node in (qf, qf_half1, qf_half2):
            node.addParam('kq', +k1)
        for node in (qd, qd_half1, qd_half2):
            node.addParam('kq', -k2)
        # Create lattice
        if start == 'drift':
            lattice.addNode(drift_half1)
            lattice.addNode(qf)
            lattice.addNode(drift2)
            lattice.addNode(qd)
            lattice.addNode(drift_half2)
        elif start == 'quad':
            lattice.addNode(qf_half1)
            lattice.addNode(drift1)
            lattice.addNode(qd)
            lattice.addNode(drift2)
            lattice.addNode(qf_half2)
        # Other
        if reverse:
            lattice.reverseOrder()
        lattice.set_fringe(fringe)
        lattice.initialize()
        for node in lattice.getNodes():
            name = node.getName()
            if 'qf' in name:
                node.setTiltAngle(+angle)
            elif 'qd' in name:
                node.setTiltAngle(-angle)
        return lattice

    def cost(kvals, correct_tunes, mass=0.93827231, energy=1):
        lattice = fodo(*kvals)
        M = transfer_matrix(lattice, mass, energy)
        return correct_phase_adv - np.degrees(get_eigtunes(M))

    correct_phase_adv = np.array([mux, muy])
    k0 = np.array([0.5, 0.5]) # ~ 80 deg phase advance
    result = opt.least_squares(cost, k0, args=(correct_phase_adv,))
    k1, k2 = result.x
    return fodo(k1, k2)
    
    
def transfer_matrix(lattice, mass, energy):
    """Shortcut to get transfer matrix from periodic lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
    
    Returns
    -------
    M : ndarray, shape (4, 4)
        Transverse transfer matrix.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    one_turn_matrix = matrix_lattice.oneTurnMatrix
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = one_turn_matrix.get(i, j)
    return M
    
    
def twiss_at_injection(lattice, mass, energy):
    """Get 2D Twiss parameters at lattice entrance.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
        
    Returns
    -------
    alpha_x, alpha_y, beta_x, beta_y : float
        2D Twiss parameters at lattice entrance.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    _, arrPosAlphaX, arrPosBetaX = matrix_lattice.getRingTwissDataX()
    _, arrPosAlphaY, arrPosBetaY = matrix_lattice.getRingTwissDataY()
    alpha_x, alpha_y = arrPosAlphaX[0][1], arrPosAlphaY[0][1]
    beta_x, beta_y = arrPosBetaX[0][1], arrPosBetaY[0][1]
    return alpha_x, alpha_y, beta_x, beta_y
    
    
def twiss_throughout(lattice, bunch):
    """Get Twiss parameters throughout lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    bunch : Bunch
        Test bunch to perform tracking.
    
    Returns
    -------
    ndarray
        Columns are: [s, nux, nuy, alpha_x, alpha_x, beta_x, beta_y]
    """
    # Extract Twiss parameters from one turn transfer matrix
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    twiss_x = matrix_lattice.getRingTwissDataX()
    twiss_y = matrix_lattice.getRingTwissDataY()
    # Unpack and convert to ndarrays
    (nux, alpha_x, beta_x), (nuy, alpha_y, beta_y) = twiss_x, twiss_y
    nux, alpha_x, beta_x = np.array(nux), np.array(alpha_x), np.array(beta_x)
    nuy, alpha_y, beta_y = np.array(nuy), np.array(alpha_y), np.array(beta_y)
    # Merge into one array
    s = nux[:, 0]
    nux, alpha_x, beta_x = nux[:, 1], alpha_x[:, 1], beta_x[:, 1]
    nuy, alpha_y, beta_y = nuy[:, 1], alpha_y[:, 1], beta_y[:, 1]
    return np.vstack([s, nux, nuy, alpha_x, alpha_y, beta_x, beta_y]).T
    
    
def get_tunes(lattice, mass, energy):
    """Compute fractional x and y lattice tunes.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].
        
    Returns
    -------
    ndarray, shape (2,)
        Array of [nux, nuy].
    """
    M = transfer_matrix(lattice, mass, energy)
    lattice_params = params_from_transfer_matrix(M)
    nux = lattice_params['frac_tune_x']
    nuy = lattice_params['frac_tune_y']
    return np.array([nux, nuy])

    
def add_node_at_start(lattice, new_node):
    """Add node as child at entrance of first node in lattice."""
    firstnode = lattice.getNodes()[0]
    firstnode.addChildNode(new_node, firstnode.ENTRANCE)


def add_node_at_end(lattice, new_node):
    """Add node as child at end of last node in lattice."""
    lastnode = lattice.getNodes()[-1]
    lastnode.addChildNode(node, lastnode.EXIT)


def add_node_throughout(lattice, new_node, position):
    """Add `new_node` as child of every node in lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        Lattice in which node will be inserted.
    new_node : NodeTEAPOT
        Node to insert.
    position : {'start', 'mid', 'end'}
        Relative location in the lattice nodes to the new node.
        
    Returns
    -------
    None
    """
    loc = {'start': AccNode.ENTRANCE, 
           'mid': AccNode.BODY, 
           'end': AccNode.EXIT}
    
    for node in lattice.getNodes():
        node.addChildNode(new_node, loc[position], 0, AccNode.BEFORE)
        
        
def get_sublattice(lattice, start_node_name=None, stop_node_name=None):
    """Return sublattice from `start_node_name` through `stop_node_name`.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        The original lattice from which to create the sublattice.
    start_node_name, stop_node_name : str
        Names of the nodes in the original lattice to use as the first and
        last node in the sublattice. 
        
    Returns
    -------
    TEAPOT_Lattice
        New lattice consisting of the specified region of the original lattice.
        Note that it is not a copy; changes to the nodes in the new lattice 
        affect the nodes in the original lattice.
    """
    if start_node_name is None:
        start_index = 0
    else:
        start_node = lattice.getNodeForName(start_node_name)
        start_index = lattice.getNodeIndex(start_node)
    if stop_node_name is None:
        stop_index = -1
    else:
        stop_node = lattice.getNodeForName(stop_node_name)
        stop_index = lattice.getNodeIndex(stop_node)
    return lattice.getSubLattice(start_index, stop_index)
        
        
def toggle_spacecharge_nodes(sc_nodes, status='off'):
    """Turn on(off) a set of space charge nodes.
    
    Parameters
    ----------
    sc_nodes : list
        List of space charge nodes. They should be subclasses of
        `SC_Base_AccNode`.
    status : {'on', 'off'}
        Whether to turn the nodes on or off.
    Returns
    -------
    None
    """
    switch = {'on':True, 'off':False}[status]
    for sc_node in sc_nodes:
        sc_node.switcher = switch


def initialize_bunch(mass, energy):
    """Create and initialize Bunch.
    
    Parameters
    ----------
    mass, energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per bunch particle.
    
    Returns
    -------
    bunch : Bunch
        A Bunch object with the given mass and kinetic energy.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(energy)
    params_dict = {'bunch': bunch}
    return bunch, params_dict
        
    
def coasting_beam(kind, nparts, twiss_params, emittances, length, mass,
                  kin_energy, intensity=0, **kws):
    """Generate bunch with no energy spread and uniform longitudinal density.
    
    Parameters
    ----------
    kind : {'kv', 'gaussian', 'waterbag'}
        The kind of distribution.
    nparts : int
        Number of macroparticles.
    twiss_params : (ax, ay, bx, by)
        2D Twiss parameters (`ax` means 'alpha x' and so on).
    emittances : (ex, ey)
        Horizontal and vertical r.m.s. emittances.
    length : float
        Bunch length [m].
    mass, kin_energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per particle.
    intensity : int
        Number of physical particles in the bunch.
    **kws
        Key word arguments for the distribution generator.
    
    Returns
    -------
    bunch : Bunch
        A Bunch object with the specified mass and kinetic energy, filled with
        particles according to the specified distribution.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(int(intensity / length) if intensity > 0 else 1)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    params_dict = {'bunch': bunch}
    constructors = {'kv':KVDist2D,
                    'gaussian':GaussDist2D,
                    'waterbag':WaterBagDist2D}
    (ax, ay, bx, by), (ex, ey) = twiss_params, emittances
    twissX = TwissContainer(ax, bx, ex)
    twissY = TwissContainer(ay, by, ey)
    dist_generator = constructors[kind](twissX, twissY, **kws)
    for i in range(nparts):
        x, xp, y, yp = dist_generator.getCoordinates()
        z = np.random.uniform(0, length)
        bunch.addParticle(x, xp, y, yp, z, 0.0)
    return bunch, params_dict
                                                                    
    
def get_coords(bunch, mm_mrad=False):
    """Extract coordinate array from bunch.
    
    Parameters
    ----------
    bunch : Bunch
        The bunch to extract the coordinates from.
    mm_mrad : bool
        Whether to convert to mm (position) and mrad (slope).
        
    Returns
    -------
    ndarray, shape (nparts, 6)
    """
    nparts = bunch.getSize()
    X = np.zeros((nparts, 6))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i),
                bunch.z(i), bunch.dE(i)]
    if mm_mrad:
        X *= 1000.
    return X
    
    
def dist_to_bunch(X, bunch, length, deltaE=0.0):
    """Fill bunch with particles.
    
    Parameters
    ----------
    X : ndarray, shape (nparts, 4)
        Transverse bunch coordinate array.
    bunch : Bunch
        The bunch to populate.
    length : float
        Bunch length. Longitudinal density is uniform.
    deltaE : float
        RMS energy spread in the bunch.
    
    Returns
    -------
    bunch : Bunch
        The modified bunch; don't really need to return this.
    """
    for (x, xp, y, yp) in X:
        z = np.random.uniform(0, length)
        dE = np.random.normal(scale=deltaE)
        bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch
    
    
def dist_from_bunch(bunch):
    """Get coordinate array from bunch.
    
    Parameters
    ----------
    bunch : Bunch
        Bunch containing `nparts` macroparticles.
    
    Returns
    -------
    X : ndarray, shape (nparts, 4)
        The transverse bunch coordinate array.
    """
    nparts = bunch.getSize()
    X = np.zeros((nparts, 4))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
    return X
    
    
def track_bunch(bunch, params_dict, lattice, nturns=1, meas_every=0,
                info='coords', progbar=True, mm_mrad=True):
    """Track a bunch through the lattice.
    
    Parameters
    ----------
    bunch : Bunch
        The bunch to track.
    params_dict : dict
        Dictionary with reference to `bunch`.
    lattice : TEAPOT_Lattice
        The lattice to track with.
    nturns : int
        Number of times to track track through the lattice.
    meas_every : int
        Store bunch info after every `dump_every` turns. If 0, no info is
        stored.
    info : {'coords', 'cov'}
        If 'coords', the transverse bunch coordinate array is stored. If `cov`,
        the transverse covariance matrix is stored.
    progbar : bool
        Whether to show tqdm progress bar.
    mm_mrad : bool
        Whether to convert from m-rad to mm-mrad.
        
    Returns
    -------
    ndarray, shape (nturns, ?, ?)
        If tracking coords, shape is (nturns, nparts, 4). If tracking
        covariance matrix, shape is (nturns, 4, 4)
    """
    info_list = []
    turns = trange(nturns) if progbar else range(nturns)
    for turn in turns:
        if meas_every > 0 and turn % meas_every == 0:
            X = get_coords(bunch, mm_mrad=mm_mrad)
            if info == 'coords':
                info_list.append(X)
            elif info == 'cov':
                info_list.append(np.cov(X.T))
        lattice.trackBunch(bunch, params_dict)
    return np.array(info_list)
