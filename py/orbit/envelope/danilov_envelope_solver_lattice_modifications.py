from envelope import DanilovEnvelopeSolver20
from envelope import DanilovEnvelopeSolver22
from orbit.envelope.danilov_envelope_solver_nodes import DanilovEnvelopeSolverNode20
from orbit.envelope.danilov_envelope_solver_nodes import DanilovEnvelopeSolverNode22
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker


def set_path_length_max(lattice, path_length_max=None):
    if path_length_max is not None:
        for node in lattice.getNodes():
            if path_length_max and node.getLength() > path_length_max:
                node.setnParts(1 + int(node.getLength() / path_length_max))
    return lattice


def set_danilov_envolope_solver_nodes(
    lattice=None, 
    path_length_max=None,
    path_length_min=None, 
    solver_node_constructor=None, 
    solver_node_constructor_args=()
):
    nodes = lattice.getNodes()
    if not nodes:
        return
    
    set_path_length_max(lattice, path_length_max)
    
    parent_nodes = []
    length_total = running_path = rest_length = 0.0
    for node in nodes:
        for part_index in range(node.getnParts()):
            part_length = node.getLength(part_index)
            if part_length > 1.0:
                message = "Warning! Node {} has length {} > 1 m.".format(node.getName(), part_length)
                message += " Space charge algorithm may be innacurate!"
                print message
            if running_path > path_length_min:
                parent_nodes.append((node, part_index, length_total, running_path))
                running_path = 0.0
            running_path += part_length
            length_total += part_length
            
    if parent_nodes:
        rest_length = length_total - parent_nodes[len(parent_nodes) - 1][2]
    else:
        rest_length = length_total
    parent_nodes.insert(0, (nodes[0], 0, 0.0, rest_length))
    
    solver_nodes = []
    for i in range(len(parent_nodes) - 1):
        (node, part_index, position, path_length) = parent_nodes[i]
        (node_next, part_index_next, position_next, path_length_next) = parent_nodes[i + 1]
        solver_node = solver_node_constructor(
            name="{}:{}:".format(node.getName(), part_index),
            *solver_node_constructor_args
        )
        solver_node.set_kick_length(path_length_next)
        solver_nodes.append(solver_node)
        node.addChildNode(solver_node, node.BODY, part_index, node.BEFORE)
        
    (node, part_index, position, path_length) = parent_nodes[len(parent_nodes)-1]
    solver_node = solver_node_constructor(
        name="{}:{}:".format(node.getName(), part_index),
        *solver_node_constructor_args
    )
    solver_node.kick_length = rest_length
    solver_nodes.append(solver_node)
    node.addChildNode(solver_node, node.BODY, part_index, node.BEFORE)
    return solver_nodes


def set_danilov_envelope_solver_nodes_20(
    lattice=None,
    path_length_max=None,
    path_length_min=1.00e-06,
    perveance=0.0,
    eps_x=20.0e-6,
    eps_y=20.0e-6,
):
    solver_nodes = set_danilov_envolope_solver_nodes(
        lattice=lattice, 
        path_length_max=path_length_max,
        path_length_min=path_length_min, 
        solver_node_constructor=DanilovEnvelopeSolverNode20, 
        solver_node_constructor_args=(perveance, eps_x, eps_y)
    )
    for solver_node in solver_nodes:
        name = "".join([solver_node.getName(), ":", "danilov_env_solver"])
        solver_node.setName(name)
    lattice.initialize()
    return solver_nodes


def set_danilov_envelope_solver_nodes_22(
    lattice=None,
    path_length_max=None,
    path_length_min=1.00e-06,
    perveance=0.0,
):
    solver_nodes = set_danilov_envolope_solver_nodes(
        lattice=lattice, 
        path_length_max=path_length_max,
        path_length_min=path_length_min, 
        solver_node_constructor=DanilovEnvelopeSolverNode22, 
        solver_node_constructor_args=(perveance,)
    )
    for solver_node in solver_nodes:
        name = "".join([solver_node.getName(), ":", "danilov_env_solver"])
        solver_node.setName(name)
    lattice.initialize()
    return solver_nodes