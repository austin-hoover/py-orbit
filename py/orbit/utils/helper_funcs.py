"""
This module contains functions for use in PyORBIT scripts.
"""

# Third party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange
# PyORBIT
from bunch import Bunch
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.matrix_lattice import MATRIX_Lattice
from orbit.bunch_generators import (
    TwissContainer, WaterBagDist2D, GaussDist2D, KVDist2D)
from orbit.utils.consts import classical_proton_radius, speed_of_light
from orbit_utils import Matrix

         
def tprint(string, indent=4):
    """Print with indent.
    
    Parameters
    ----------
    string : str
        String to print.
    indent : int
        Number of spaces of indent.
    
    Returns
    -------
    None
    """
    print indent*' ' + str(string)
    
    
def step_func(x):
    "Heaviside step function."
    return x if x >= 0 else 0
    
    
def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def normalize(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x/la.norm(x), 1, X)


def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())
    
    
def rand_rows(X, n):
    """Return n random elements of X."""
    Xsamp = np.copy(X)
    if n < X.shape[0]:
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp
    
    
def rotation_matrix(phi):
    """2D rotation matrix (cw)."""
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])


def rotation_matrix_4D(phi):
    """Rotate [x, x', y, y'] cw in the x-y plane."""
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, 0, S, 0], [0, C, 0, S], [-S, 0, C, 0], [0, -S, 0, C]])


def phase_adv_matrix(mu1, mu2):
    """Rotate [x, x'] by mu1 and [y, y'] by mu2, both clockwise."""
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(mu1)
    R[2:, 2:] = rotation_matrix(mu2)
    return R
    
    
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
    
    
def get_Brho(mass, kin_energy):
    """Compute magnetic rigidity (B rho = p / c)."""
    pc = np.sqrt(kin_energy * (kin_energy + 2 * mass))
    return 1e9 * (pc / speed_of_light)
    
    
def tilt_elements_containing(lattice, key, angle):
    """Tilt all elements with `key` in their name. Only used this once... may
    delete.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        The lattice containing the elements.
    key : str
        Key word which contained in element names to be tilted. For example:
        'qh' could tilt all horizontal quads.
    angle : float
        Tilt angle [rad].
    
    Returns
    -------
    None
    """
    for node in lattice.get_nodes_containing(key):
        node.setTiltAngle(angle)


def fodo_lattice(mux, muy, L, fill_fac, angle=0, start='drift', fringe=False,
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
        half_nodes = (drift_half1, drift_half2, qf_half1, qf_half2, qd_half1,
                      qd_half2)
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
        tilt_elements_containing(lattice, 'qf', +angle)
        tilt_elements_containing(lattice, 'qd', -angle)
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
        
    Returns
    -------
    TEAPOT_Lattice
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
    

def params_from_transfer_matrix(M):
    """Return dictionary of lattice parameters from a transfer matrix.
    
    Method is taken from `py/orbit/matrix_lattice/MATRIX_Lattice.py`.
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        A transfer matrix.
        
    Returns
    -------
    lattice_params : dict
        Dictionary with the following keys: 'frac_tune_x', 'frac_tune_y',
        'alpha_x', 'alpha_y', 'beta_x', 'beta_y', 'gamma_x', 'gamma_y'.
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

    
def add_node_at_start(lattice, node):
    """Add node at entrance of first node in lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        Lattice in which node will be inserted.
    node : NodeTEAPOT
        Node to insert.
        
    Returns
    -------
    None
    """
    firstnode = lattice.getNodes()[0]
    firstnode.addChildNode(node, firstnode.ENTRANCE)


def add_node_at_end(lattice, node):
    """Add node at end of the last node in lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice
        Lattice in which node will be inserted.
    node : NodeTEAPOT
        Node to insert.
        
    Returns
    -------
    None
    """
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
    loc = {
        'start': AccNode.ENTRANCE, 
        'mid': AccNode.BODY, 
        'end': AccNode.EXIT
    }
    for node in lattice.getNodes():
        node.addChildNode(new_node, loc[position], 0, AccNode.BEFORE)
        
        
def is_stable(M, tol=1e-5):
    """Determine transfer matrix stability.
    
    Parameters
    ----------
    M : ndarray, shape (n, n)
        A transfer matrix.
    tol : float
        The matrix is stable if all eigenvalue norms are in the range [1 - tol,
        1 + tol].
    
    Returns
    -------
    bool
    """
    for eigval in la.eigvals(M):
        if abs(la.norm(eigval) - 1) > tol:
            return False
    return True
    
    
def get_eigtunes(M):
    """Compute transfer matrix eigentunes -- cos(Re[eigenvalue]).
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        A transfer matrix.

    Returns
    -------
    ndarray, shape (2,)
        Eigentunes for each mode.
    """
    return np.arccos(la.eigvals(M).real)[[0, 2]]
    

def unequal_eigtunes(M, tol=1e-5):
    """Return True if the eigentunes of the transfer matrix are the same.
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        A transfer matrix.
    tol : float
        Eigentunes are equal if abs(mu1 - mu2) > tol.

    Returns
    -------
    bool
    """
    mu1, mu2 = get_eigtunes(M)
    return abs(mu1 - mu2) > tol
        
        
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
    bunch.macroSize(int(intensity / length))
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
    
    
def dist_to_bunch(X, bunch, length):
    """Fill bunch with particles (coasting).
    
    Parameters
    ----------
    X : ndarray, shape (nparts, 4)
        Transverse bunch coordinate array.
    bunch : Bunch
        The bunch to populate.
    length : float
        Bunch length. Axial density is uniform and there is no energy spread.
    
    Returns
    -------
    bunch : Bunch
        The modified bunch; don't really need to return this.
    """
    for (x, xp, y, yp) in X:
        z = np.random.uniform(0, length)
        bunch.addParticle(x, xp, y, yp, z, 0.)
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
