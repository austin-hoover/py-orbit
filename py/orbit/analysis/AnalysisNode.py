from bunch import Bunch
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
from analysis import Stats

class AnalysisNode(DriftTEAPOT):
    
    def __init__(self, file_path, position=0.0, name='analysis_no_name'):
        DriftTEAPOT.__init__(self, name)
        self.stats = Stats(file_path)
        self.setType('analysis')
        self.setLength(0.0)
        self.position = position

    def track(self, params_dict):
        length = self.getLength(self.getActivePartIndex())
        self.stats.write(self.position, params_dict['bunch'])
            
    def setPosition(self, position):
        self.position = position
        
    def close(self):
        self.stats.close()
