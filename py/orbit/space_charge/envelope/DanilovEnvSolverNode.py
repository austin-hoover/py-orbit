from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker
from orbit.space_charge.scAccNodes import SC_Base_AccNode
from orbit.utils import orbitFinalize


class DanilovEnvSolverNode(SC_Base_AccNode):
    def __init__(self, sc_calculator, name="danilov_env_solver"):
        SC_Base_AccNode.__init__(self, sc_calculator, name)
        self.setType("DanilovEnvSolver")
        self.sc_calculator = sc_calculator

    def track(self, params_dict):
        if not self.switcher:
            return
        bunch = params_dict["bunch"]
        self.sc_calculator.trackBunch(bunch, self.sc_length)
