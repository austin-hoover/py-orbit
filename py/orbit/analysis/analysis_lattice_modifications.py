"""
Module for inserting analysis or monitor nodes into the lattice.
"""

from orbit.utils import orbitFinalize
from orbit.lattice import AccLattice, AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
from AnalysisNode import AnalysisNode
from MonitorNode import OnePartMonitorNode, EnvMonitorNode


def idx_pos_list(nodes, min_sep=1e-5):
    """Create a list of (node, part_idx, position)."""
    nodes_arr = []
    position = running_length = 0.
    for node in nodes:
        for i in range(node.getnParts()):
            part_length = node.getLength(i)
            if running_length > min_sep:
                nodes_arr.append((node, i, position))
                temp = 0.
            running_length += part_length
            position += part_length
    nodes_arr.insert(0, (nodes[0], 0, 0.))
    return nodes_arr


def add_analysis_nodes_at_centers(lattice, output_dir):
    for node in lattice.getNodes():
        position = lattice.getNodePositionsDict()[node][0]
        analysis_node = AnalysisNode(output_dir, position)
        node.addChildNode(analysis_node, node.ENTRANCE)
    lattice.initialize()
    
    
def add_monitor_nodes_at_centers(lattice, filename, constructor):
    for node in lattice.getNodes():
        position = lattice.getNodePositionsDict()[node][0]
        monitor_node = constructor(filename, position)
        node.addChildNode(monitor_node, node.ENTRANCE)
    lattice.initialize()
    
        
def add_analysis_nodes(lattice, output_dir, min_sep=0.00001):
    """Add analysis nodes at start of each node in the lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice object
        The lattice to insert the nodes into.
    output_dir : str
        The directory to store the output files.
    min_sep : float
        The minimum separation between the analysis nodes.
        
    Returns
    -------
    analysis_nodes : List[AccNodes]
        List of the inserted analysis nodes.
    """
    nodes = lattice.getNodes()
    if len(nodes) == 0:
        return
        
    analysis_nodes = []
    for (node, idx, position) in idx_pos_list(nodes, min_sep):
        name = ''.join([node.getName(), ':', str(idx), ':'])
        analysis_node = AnalysisNode(output_dir, position, name)
        node.addChildNode(analysis_node, AccNode.BODY, idx, AccNode.BEFORE)
        analysis_nodes.append(analysis_node)
    return analysis_nodes
    
    
def add_monitor_nodes(lattice, filename, constructor, min_sep=0.00001):
    """Add one particle monitor nodes at start of each node in lattice.
    
    Parameters
    ----------
    lattice : TEAPOT_Lattice object
        The lattice to insert the nodes into.
    filename : str
        The file to store the particle coordinates.
    constructor : AccNode object
        The constructor for the monitor node which is called as
        monitor_node = Constructor(filename, position, name).
    min_sep : float
        The minimum separation between the monitor nodes.
        
    Returns
    -------
    monitor_nodes : List[AccNodes]
        List of the inserted monitor nodes.
    """
    nodes = lattice.getNodes()
    if len(nodes) == 0:
        return

    monitor_nodes = []
    for (node, idx, position) in idx_pos_list(nodes, min_sep):
        name = ''.join(['monitor_', str(idx)])
        tilt = node.getAllChildren()[0].getTiltAngle()
        monitor_node = constructor(filename, position, name, tilt)
        node.addChildNode(monitor_node, AccNode.BODY, idx, AccNode.BEFORE)
        monitor_nodes.append(monitor_node)
    return monitor_nodes
