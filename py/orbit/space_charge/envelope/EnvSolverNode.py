from orbit.lattice import AccLattice, AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.space_charge.scAccNodes import SC_Base_AccNode
from orbit.utils import orbitFinalize

class EnvSolverNode(SC_Base_AccNode):
    
    def __init__(self, sc_calculator, name='envsolver'):
        """Class implementation of envelope solver node."""
        SC_Base_AccNode.__init__(self, sc_calculator, name)
        self.setType('envsolver')
        self.sc_calculator = sc_calculator

    def track(self, paramsDict):
        if not self.switcher:
            return
        bunch = paramsDict['bunch']
        self.sc_calculator.trackBunch(bunch, self.sc_length)
