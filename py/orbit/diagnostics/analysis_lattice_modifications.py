"""Module for inserting analysis nodes into the lattice."""
from analysis_node import AnalysisNode 
from analysis_node import BunchMonitorNode
from analysis_node import BunchStatsNode
from analysis_node import DanilovEnvelopeBunchMonitorNode
from analysis_node import WireScannerNode
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT


NAMETAGS = {
    BunchMonitorNode: 'bunch_monitor',
    BunchStatsNode: 'bunch_stats',
    DanilovEnvelopeBunchMonitorNode: 'danilov_envelope_bunch_monitor',
    WireScannerNode: 'wire-scanner',
}


def node_index_position_list(lattice, min_sep=1e-5):
    items = []
    position = running_length = 0.
    for node in lattice.getNodes():
        for part_index in range(node.getnParts()):
            part_length = node.getLength(part_index)
            if running_length > min_sep:
                items.append((node, part_index, position))
            running_length += part_length
            position += part_length
    first_node = lattice.getNodes()[0]
    items.insert(0, (first_node, 0, 0.))
    return items


def add_analysis_node(constructor, lattice, parent_node, **constructor_kws):
    """Create/insert analysis node at one point in the lattice.

    Parameters
    ----------
    constructor : AnalysisNode subclass
        This is called to create the analysis node.
    lattice : AccLattice
        The lattice in which to insert the node. This is provided to get the position
        of the parent node.
    parent_node : AccNode or str
        The parent node of the analysis node. Can also just provide its name.
    dense: bool
        Whether to insert at every part of every node rather than just at every node.
    **constructor_kws
        Key word arguments passed to the analysis node constructor.
    """
    if type(parent_node) is str:
        parent_node = lattice.getNodeForName(parent_node)
    analysis_node = constructor(**constructor_kws)
    analysis_node.setName('{}:{}'.format(parent_node.getName(), NAMETAGS[constructor]))
    analysis_node.set_position(lattice.getNodePositionsDict()[parent_node][0])
    analysis_node.setTiltAngle(-parent_node.getTiltAngle())
    parent_node.addChildNode(analysis_node, parent_node.ENTRANCE)
    return analysis_node


def add_analysis_nodes(constructor, lattice, dense=False, min_sep=1e-5, **constructor_kws):
    """Create/insert analysis nodes throughout the lattice.

    Parameters
    ----------
    constructor : AnalysisNode subclass
        This is called to create the analysis node.
    lattice : AccLattice
        The lattice in which to insert the nodes.
    dense: bool
        Whether to insert at every part of every node rather than just at every node.
    **constructor_kws
        Key word arguments passed to the analysis node constructor.
    """
    if not dense:
        analysis_nodes = []
        for node in lattice.getNodes():
            analysis_node = add_analysis_node(constructor, lattice, node, **constructor_kws)
            analysis_nodes.append(analysis_node)
        return analysis_nodes

    analysis_nodes = []
    for (parent_node, part_index, position) in node_index_position_list(lattice, min_sep):
        analysis_node = constructor(**constructor_kws)
        analysis_node.setName('{}:{}:{}'.format(parent_node.getName(), NAMETAGS[constructor], part_index))
        analysis_node.set_position(position)
        analysis_node.setTiltAngle(-parent_node.getAllChildren()[0].getTiltAngle())
        parent_node.addChildNode(analysis_node, AccNode.BODY, part_index, AccNode.BEFORE)
        analysis_nodes.append(analysis_node)
    return analysis_nodes