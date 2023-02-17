"""
Module. Includes functions that will modify the accelerator lattice by inserting the one diagnostics node accelerator node.
"""
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
from orbit.utils import orbitFinalize
from TeapotDiagnosticsNode import TeapotMomentsNode
from TeapotDiagnosticsNode import TeapotMomentsNodeSetMember
from TeapotDiagnosticsNode import TeapotStatLatsNode
from TeapotDiagnosticsNode import TeapotStatLatsNodeSetMember


def addTeapotDiagnosticsNode(lattice, position, diagnostics_node):
    """
    It will put one Teapot diagnostics node in the lattice 
    """
    length_tollerance = 0.0001
    lattice.initialize()
    position_start = position
    position_stop = position + diagnostics_node.getLength()
    diagnostics_node.setPosition(position)
    diagnostics_node.setLatticeLength(lattice.getLength())
    (node_start_ind,node_stop_ind,z,ind) = (-1,-1, 0., 0)
    for node in lattice.getNodes():
        if(position_start >= z and position_start <= z + node.getLength()):
            node_start_ind = ind
        if(position_stop >= z and position_stop <= z + node.getLength()):
            node_stop_ind = ind
        ind += 1
        z += node.getLength()
    #-------now we check that between start and end we have only non-modified drift elements
    #-------if the space charge was added first - that is a problem. The collimation should be added first.
    for node in lattice.getNodes()[node_start_ind:node_stop_ind+1]:
        #print "debug node=",node.getName()," type=",node.getType()," L=",node.getLength()
        if(not isinstance(node,DriftTEAPOT)):
            print "Non-drift node=",node.getName()," type=",node.getType()," L=",node.getLength()
            orbitFinalize("We have non-drift element at the place of the diagnostics! Stop!")

    # make array of nodes from diagnostics in the center and possible two drifts if their length is more than length_tollerance [m]
    nodes_new_arr = [diagnostics_node,]
    drift_node_start = lattice.getNodes()[node_start_ind]
    drift_node_stop = lattice.getNodes()[node_stop_ind]	
    #------now we will create two drift nodes: before the diagnostics and after
    #------if the length of one of these additional drifts less than length_tollerance [m] we skip this drift 
    if(position_start > lattice.getNodePositionsDict()[drift_node_start][0] +  length_tollerance):
        drift_node_start_new = DriftTEAPOT(drift_node_start.getName())
        drift_node_start_new.setLength(position_start - lattice.getNodePositionsDict()[drift_node_start][0])
        nodes_new_arr.insert(0,drift_node_start_new)
    if(position_stop < lattice.getNodePositionsDict()[drift_node_stop][1] - length_tollerance):
        drift_node_stop_new = DriftTEAPOT(drift_node_stop.getName())
        drift_node_stop_new.setLength(lattice.getNodePositionsDict()[drift_node_stop][1] - position_stop)
        nodes_new_arr.append(drift_node_stop_new)
    #------ now we will modify the lattice by replacing the found part with the new nodes
    lattice.getNodes()[node_start_ind:node_stop_ind+1] = nodes_new_arr
    # initialize the lattice
    lattice.initialize()

def addTeapotDiagnosticsNodeAsChild(lattice, AccNode, diagnostics_node):
    AccNode.addChildNode(diagnostics_node, AccNode.ENTRANCE,0,AccNode.BEFORE)
    lattice.initialize()

def addTeapotStatLatsNodeSet(lattice, filename):
    """
    It will put one Teapot statlats node at start of each node in lattice
    """
    file_out = open(filename, "w")
    nodesetcontroller = diagnosticsNodeSetController(file_out, "StatLats Set Controller")
    lattice.initialize()
    for node in lattice.getNodes():
        position = lattice.getNodePositionsDict()[node][0]
        diagnostics_node = TeapotStatLatsNodeSetMember(file_out)
        diagnostics_node.setPosition(position)
        diagnostics_node.setLatticeLength(lattice.getLength())
        node.addChildNode(diagnostics_node, node.BODY)
        nodesetcontroller._nodelist.append(diagnostics_node)
    return nodesetcontroller

def addTeapotMomentsNodeSet(lattice, filename, order, nodispersion = True, emitnorm = False):
    """
    It will put one Teapot statlats node at start of each node in lattice
    """
    file_out = open(filename, "w")
    nodesetcontroller = diagnosticsNodeSetController(file_out, "Moment Set Controller")
    lattice.initialize()
    print 'In lattice modification no dispersion is ', nodispersion
    for node in lattice.getNodes():
        position = lattice.getNodePositionsDict()[node][0]
        diagnostics_node = TeapotMomentsNodeSetMember(file_out, order, nodispersion, emitnorm,  "moments")
        diagnostics_node.setPosition(position)
        diagnostics_node.setLatticeLength(lattice.getLength())
        node.addChildNode(diagnostics_node, node.BODY)
        nodesetcontroller._nodelist.append(diagnostics_node)
    return nodesetcontroller

class diagnosticsNodeSetController:
    """
        This class keeps lists of moment nodes and acts as a controller
    """

    def __init__(self, file = None, name = "Diagnostic Set Controller"):
        self._nodelist = []
        self._file = file

    def activate(self):
        for node in self._nodelist:
            node.activate()

    def deactivate(self):
        for node in self._nodelist:
            node.deactivate()

    def resetFile(self, filename):
        self._file.close()
        file_out = open(filename, "w")
        self._file = file_out
        for node in self._nodelist:
            node.resetFile(self._file)


def add_diagnostics_node_as_child(
    diagnostics_node, 
    parent_node=None, 
    part_index=0, 
    place_in_part=AccActionsContainer.BEFORE,
):
    """Add diagnostics node as child at entrance of parent.

    The tilt angle is set to the negative of the parent node tilt angle. 
    (PyORBIT handles tilted nodes by rotating the bunch coordinates at the
    node entrance/exit.)
    """
    diagnostics_node.setName(
        '{}:{}:{}'.format(
            parent_node.getName(), 
            diagnostics_node.getName(), 
            part_index,
        )
    )
    try:
        diagnostics_node.setTiltAngle(-parent_node.getTiltAngle())
    except:
        pass
    parent_node.addChildNode(
        diagnostics_node, 
        parent_node.ENTRANCE, 
        part_index=part_index,
        place_in_part=AccActionsContainer.BEFORE,
    )
    return diagnostics_node

    
def node_index_position_list(lattice, min_sep=1e-5):
    """Return list of (node, index, position)."""
    items = []
    position = running_length = 0.0
    for node in lattice.getNodes():
        for part_index in range(node.getnParts()):
            part_length = node.getLength(part_index)
            if running_length > min_sep:
                items.append((node, part_index, position))
            running_length += part_length
            position += part_length
    first_node = lattice.getNodes()[0]
    items.insert(0, (first_node, 0, 0.0))
    return items


def add_diagnostics_node(lattice, diagnostics_node, position=0.0):
    """Add a diagnostics node at one position the lattice."""
    length_tollerance = 0.0001
    lattice.initialize()
    position_start = position
    position_stop = position + diagnostics_node.getLength()
    (node_start_ind, node_stop_ind, z, ind) = (-1, -1, 0.0, 0)
    nodes = lattice.getNodes()
    for node in nodes:
        if position_start >= z and position_start <= z + node.getLength():
            node_start_ind = ind
        if position_stop >= z and position_stop <= z + node.getLength():
            node_stop_ind = ind
        ind += 1
        z += node.getLength()
    # Now we check that we have only non-modified drift elements between start
    # and end. If the space charge was added first - that is a problem. The 
    # collimation should be added first.
    for node in nodes[node_start_ind : node_stop_ind + 1]:
        if not isinstance(node, DriftTEAPOT):
            print(
                "Non-drift node=",
                node.getName(),
                " type=",
                node.getType(),
                " L=",
                node.getLength(),
            )
            orbitFinalize(
                "We have non-drift element at the place of the diagnostics! Stop!"
            )
    # Make array of nodes from diagnostics in the center and possible two drifts if
    # their length is more than length_tollerance [m].
    nodes_new_arr = [diagnostics_node]
    drift_node_start = nodes[node_start_ind]
    drift_node_stop = nodes[node_stop_ind]
    # Create two drift nodes: before the diagnostics and after. Skip this drift if
    # either of these additional drifts is less than length_tollerance.   
    if position_start > lattice.getNodePositionsDict()[drift_node_start][0] + length_tollerance:
        drift_node_start_new = DriftTEAPOT(drift_node_start.getName())
        drift_node_start_new.setLength(position_start - lattice.getNodePositionsDict()[drift_node_start][0])
        nodes_new_arr.insert(0, drift_node_start_new)
    if position_stop < lattice.getNodePositionsDict()[drift_node_stop][1] - length_tollerance:
        drift_node_stop_new = DriftTEAPOT(drift_node_stop.getName())
        drift_node_stop_new.setLength(lattice.getNodePositionsDict()[drift_node_stop][1] - position_stop)
        nodes_new_arr.append(drift_node_stop_new)
    # Replace the found part with the new nodes.
    nodes[node_start_ind : node_stop_ind + 1] = nodes_new_arr
    lattice.initialize()
