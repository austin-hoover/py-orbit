"""
Module to implement the AnalysisNode class.
"""

# 3rd party
import numpy as np
# PyORBIT
from bunch import Bunch
from orbit.analysis import Stats
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject


def get_coords(bunch, mm_mrad=False):
    """Return the transverse coordinate array from the bunch."""
    nparts = bunch.getSize()
    X = np.zeros((nparts, 4))
    for i in range(nparts):
        X[i] = [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
    if mm_mrad:
        X *= 1000
    return X
        
        
class EnvBunch:
    """Container for a bunch which stores the envelope parameters and test
    bunch coordinates"""
    def __init__(self, X):
        (a, ap, e, ep), (b, bp, f, fp) = X[:2]
        self.env_params = np.array([a, b, ap, bp, e, f, ep, fp])
        self.testbunch_coords = None
        if X.shape > 2:
            self.testbunch_coords = X[2:]
        
        
class AnalysisNode(DriftTEAPOT):
    """Node to store beam parameters.
    
    The beam parameters could be the coordinate array, twiss parameters,
    moments, or envelope parameters. The node stores a list of these
    parameters and appends to the list each time `track` is called.
    
    Attributes
    ----------
    position : float
        The s position of the node [m].
    data : list
        Each element in the list contains some beam data. A new element is
        added each time `track` is called.
    kind : str
        The kind of analysis node. The options are:
        'env_monitor'
            Stores the envelope parameters, which are contained in the
            first two bunch particles, as well as any test particles in
            the bunch.
        'bunch_monitor'
            Stores the bunch coordinate array.
        'bunch_stats'
            Stores the bunch moments and twiss parameters.
    """
    def __init__(self, position, kind, name='analysis', mm_mrad=True):
        DriftTEAPOT.__init__(self, name)
        self.position = position
        self.setLength(0.0)
        self.kind = kind
        self.mm_mrad = mm_mrad
        self.data = []
    
    def track(self, params_dict):
        """Store the beam data."""
        X = get_coords(params_dict['bunch'], self.mm_mrad)
        if self.kind == 'env_monitor':
            _data = EnvBunch(X)
        elif self.kind == 'bunch_monitor':
            _data = X
        elif self.kind == 'bunch_stats':
            _data = Stats(X)
        self.data.append(_data)
        
    def clear_data(self):
        """Delete all data stored in the node."""
        self.data = []
        
    def get_data(self, dtype, turn=0):
        """Extract the data from the node.
        
        dtype : str
            'env_params': the envelope parameters
            'testbunch_coords': the test bunch coordinates
            'bunch_coords': the bunch coordinates
            'bunch_twiss': the test bunch Twiss parameters
            'bunch_moments': the bunch moments
        turn : int or str
            If an int, `turn` is the turn number of position in data list.
            Choosing `all_turns` will return the the data for all turns in
            one array. This can probably be combined so that the data from
            turn `i` to turn `j` are returned.
        """
        if type(turn) is int:
            _data = self.data[turn]
            if dtype == 'env_params':
                return _data.env_params
            elif dtype == 'testbunch_coords':
                return _data.testbunch_coords
            elif dtype == 'bunch_coords':
                return _data
            elif dtype == 'bunch_twiss':
                return _data.twiss
            elif dtype == 'bunch_moments':
                return _data.moments
            elif dtype == 'bunch_cov':
                return _data.Sigma
        elif turn == 'all_turns':
            if dtype == 'env_params':
                return np.array([d.env_params for d in self.data])
            elif dtype == 'testbunch_coords':
                return np.array([d.testbunch_coords for d in self.data])
            elif dtype == 'bunch_coords':
                return np.array(self.data)
            elif dtype == 'bunch_twiss':
                return np.array([d.twiss for d in self.data])
            elif dtype == 'bunch_twiss':
                return np.array([d.twiss for d in self.data])
            elif dtype == 'bunch_moments':
                return np.array([d.moments for d in self.data])

    
def get_analysis_nodes_data(analysis_nodes, dtype, turn=0):
    """Return an array of the data from every node in the list.
    
    This is used when the beam is tracked once throught the lattice and
    we want the data as a function of s.
    """
    if dtype == 'position':
        return np.array([node.position for node in analysis_nodes])
    return np.array([node.get_data(dtype, turn) for node in analysis_nodes])
    
    
def clear_analysis_nodes_data(analysis_nodes):
    """Delete the data stored in the nodes."""
    for node in analysis_nodes:
        node.clear_data()


class WireScannerNode(DriftTEAPOT):
    """Simple node to measure <x^2>, <y^2>, and <xy>.
    
    To do: add measurement error.
    """
    def __init__(self, name='ws'):
        DriftTEAPOT.__init__(self, name)
        self.x2 = self.y2 = self.xy = 0.0

    def track(self, params_dict):
        X = get_coords(params_dict['bunch'])
        Sigma = np.cov(X.T)
        self.x2, self.y2, self.xy = Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]
