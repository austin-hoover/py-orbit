"""
Module for inserting analysis nodes into the lattice.
"""

# 3rd party
import numpy as np
# PyORBIT
from orbit.lattice import (
    AccLattice,
    AccNode,
    AccActionsContainer,
    AccNodeBunchTracker)
from orbit.teapot import DriftTEAPOT
from orbit.utils import orbitFinalize
from orbit.analysis import AnalysisNode


def idx_pos_list(nodes, min_sep=1e-5):
    """Return a list of (node, part_idx, position)."""
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


def add_analysis_nodes(lattice, kind='env_monitor', min_sep=1e-5):
    """Add analysis node at the start of the BODY component of each node.

    Parameters
    ----------
    lattice : TEAPOT_Lattice object
        The lattice to insert the nodes into.
    kind : str
        This determines the type of data stored in the node. Options are:
        'env_monitor' -- the envelope parameters
        'bunch_monitor' -- the bunch coordinate array.
        'stats' -- the bunch statistics.
    min_sep : float
        The minimum distance between monitor nodes [m].
    """
    nodes, analysis_nodes = lattice.getNodes(), []
    if len(nodes) == 0:
        return
    def add_analysis_node(parent_node, idx, position):
        name = ''.join([parent_node.getName(), ':', kind, '_monitor_', str(idx)])
        analysis_node = AnalysisNode(position, kind, name)
        tilt_angle = parent_node.getAllChildren()[0].getTiltAngle()
        analysis_node.setTiltAngle(-tilt_angle)
        parent_node.addChildNode(analysis_node, AccNode.BODY,
                                 idx, AccNode.BEFORE)
        return analysis_node

    analysis_nodes = []
    for (node, idx, pos) in idx_pos_list(nodes, min_sep):
        analysis_nodes.append(add_analysis_node(node, idx, pos))

    # Add node at exit of lattice
    last_node, last_position = lattice.getNodes()[-1], lattice.getLength()
    analysis_node = add_analysis_node(last_node, -1, last_position)
    analysis_nodes.append(analysis_node)
    return analysis_nodes
