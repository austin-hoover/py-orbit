import numpy as np

from bunch import Bunch
from analysis import bunch_coord_array
from analysis import BunchStats
from analysis import DanilovEnvelopeBunch
from orbit.teapot import DriftTEAPOT


class AnalysisNode(DriftTEAPOT):
    
    def __init__(self, name):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        self.position = None
        self.kind = None
        self.data = []
        
    def set_position(self, position):
        self.position = position
        
    def track(self, params_dict):
        return
        
    def get_data(self, turn='all'):
        if turn == 'all':
            return self.data
        return self.data[turn]
    
    def clear_data(self):
        self.data = []
    
                    
class BunchMonitorNode(AnalysisNode):
    
    def __init__(self, name='bunch_monitor', mm_mrad=False, transverse_only=True):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        self.transverse_only = transverse_only
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, self.transverse_only)
        self.data.append(X)
        
    
class BunchStatsNode(AnalysisNode):
    
    def __init__(self, name='bunch_stats', mm_mrad=False):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
        Sigma = np.cov(X.T)
        self.data.append(BunchStats(Sigma))
        
        
class DanilovEnvelopeBunchMonitorNode(AnalysisNode):
    
    def __init__(self, name='envelope_monitor', mm_mrad=False):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
        self.data.append(DanilovEnvelopeBunch(X))