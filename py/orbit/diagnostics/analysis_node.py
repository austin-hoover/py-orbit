import numpy as np

from bunch import Bunch
from analysis import bunch_coord_array
from analysis import BunchStats
from analysis import DanilovEnvelopeBunch
from orbit.teapot import DriftTEAPOT


class AnalysisNode(DriftTEAPOT):
    
    def __init__(self, name, skip=0):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        self.position = None
        self.data = []
        self.turn = 0
        self.turns_stored = []
        self.skip = skip
        self.active = True
        
    def set_position(self, position):
        self.position = position
                
    def get_data(self, turn='all'):
        if turn == 'all':
            return self.data
        return self.data[turn]
    
    def clear_data(self):
        self.data = []
        
    def should_store(self):
        return self.turn % (self.skip + 1) == 0
    
    def set_active(self, active):
        self.active = active
    
                    
class BunchMonitorNode(AnalysisNode):
    
    def __init__(self, name='bunch_monitor', mm_mrad=False, transverse_only=True, skip=0):
        AnalysisNode.__init__(self, name, skip)
        self.mm_mrad = mm_mrad
        self.transverse_only = transverse_only
        
    def track(self, params_dict):
        if self.should_store() and self.active:
            bunch = params_dict['bunch']
            X = bunch_coord_array(bunch, self.mm_mrad, self.transverse_only)
            self.data.append(X)
            self.turns_stored.append(self.turn)
        self.turn += 1
        
    
class BunchStatsNode(AnalysisNode):
    
    def __init__(self, name='bunch_stats', mm_mrad=False, skip=0):
        AnalysisNode.__init__(self, name, skip)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        if self.should_store() and self.active:
            bunch = params_dict['bunch']
            X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
            Sigma = np.cov(X.T)
            bunch_stats = BunchStats(Sigma)
            self.data.append(bunch_stats)
            self.turns_stored.append(self.turn)
        self.turn += 1
        
        
class DanilovEnvelopeBunchMonitorNode(AnalysisNode):

    def __init__(self, name='envelope_monitor', mm_mrad=False, skip=0):
        AnalysisNode.__init__(self, name, skip)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        if self.should_store() and self.active:
            bunch = params_dict['bunch']
            X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
            env_bunch = DanilovEnvelopeBunch(X)
            self.data.append(env_bunch)
            self.turns_stored.append(self.turn)
        self.turn += 1