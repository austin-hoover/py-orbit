from envelope import DanilovEnvelopeSolver20
from envelope import DanilovEnvelopeSolver22
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker
from orbit.utils import orbitFinalize


class DanilovEnvelopeSolverNode(AccNodeBunchTracker):
    def __init__(self, solver=None, name="danilov_envelope_solver", kick_length=0.0):
        AccNodeBunchTracker.__init__(self, name)
        self.setType("DanilovEnvSolver")
        self.solver = solver
        self.kick_length = kick_length
        self.active = True

    def set_kick_length(self, kick_length):
        self.kick_length = kick_length

    def set_active(self, active):
        self.active = active

    def isRFGap(self):
        return False

    def trackDesign(self, params_dict):
        pass

    def track(self, params_dict):
        if not self.active:
            return
        bunch = params_dict["bunch"]
        self.solver.trackBunch(bunch, self.kick_length)


class DanilovEnvelopeSolverNode20(DanilovEnvelopeSolverNode):
    def __init__(
        self,
        perveance=0.0,
        eps_x=1.0e-6,
        eps_y=1.0e-6,
        kick_length=0.0,
        name="danilov_envelope_solver",
    ):
        DanilovEnvelopeSolverNode.__init__(
            self,
            solver=DanilovEnvelopeSolver20(perveance, eps_x, eps_y),
            name=name,
            kick_length=kick_length,
        )

    def set_perveance(self, perveance):
        self.solver.setPerveance(perveance)

    def set_emittance(self, eps_x, eps_y):
        self.solver.setEmittance(eps_x, eps_y)


class DanilovEnvelopeSolverNode22(DanilovEnvelopeSolverNode):
    def __init__(self, perveance=0.0, kick_length=0.0, name="danilov_envelope_solver_22"):
        DanilovEnvelopeSolverNode.__init__(
            self,
            solver=DanilovEnvelopeSolver22(perveance),
            name=name,
            kick_length=kick_length,
        )

    def set_perveance(self, perveance):
        self.solver.setPerveance(perveance)
