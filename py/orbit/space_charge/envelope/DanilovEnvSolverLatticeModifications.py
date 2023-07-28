from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.scLatticeModifications import setSC_General_AccNodes
from spacecharge import DanilovEnvSolver


def setDanilovEnvSolverNodes(lattice=None, perveance=0.0, max_sep=None, min_sep=1.00e-06):
    """Place a set of envelope solver nodes into the lattice.

    The method will place the set into the lattice as child nodes of
    the first level accelerator nodes. The nodes will be inserted at
    the beginning of a particular part of the first level AccNode
    element.

    Parameters
    ----------
    lattice : AccLattice
        Lattice in which to insert the nodes.
    perveance : float
        Dimensionless beam perveance.
    max_sep, min_sep : float
        Maximum/minimum separation between the nodes.

    Returns
    -------
    list[DanilovEnvSolverNode]
        List of the inserted envelope solver nodes.
    """
    for node in lattice.getNodes():
        if max_sep and node.getLength() > max_sep:
            node.setnParts(1 + int(node.getLength() / max_sep))

    solver = DanilovEnvSolver(perveance)
    solver_nodes = setSC_General_AccNodes(lattice, min_sep, solver, DanilovEnvSolverNode)
    for solver_node in solver_nodes:
        name = "".join([solver_node.getName(), ":", "envsolver"])
        solver_node.setName(name)
    lattice.initialize()
    return solver_nodes
