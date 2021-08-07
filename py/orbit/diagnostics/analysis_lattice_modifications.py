"""Module for inserting analysis nodes into the lattice."""
from analysis_node import AnalysisNode 
from analysis_node import BunchMonitorNode
from analysis_node import BunchStatsNode
from analysis_node import DanilovEnvelopeBunchMonitorNode
from analysis_node import WireScannerNode
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT


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


class AnalysisNodeInserter:
    
    def __init__(self, constructor, **constructor_kws):
        self.constructor = constructor
        self.constructor_kws = constructor_kws
        if self.constructor_kws is None:
            self.constructor_kws = dict()
        self.nametag = 'analysis'
        if self.constructor == BunchMonitorNode:
            self.nametag = 'bunch_monitor'
        elif self.constructor == BunchStatsNode:
            self.nametag == 'bunch_stats'
        elif self.constructor == DanilovEnvelopeBunchMonitorNode:
            self.nametag = 'danilov_envelope_bunch_monitor'
        elif self.constructor == WireScannerNode:
            self.nametag = 'wire-scanner'
            
    def create_node(self):
        return self.constructor(**self.constructor_kws)
        
    def insert_at_entrance(self, lattice, parent_node):
        if type(parent_node) is str:
            parent_node = lattice.getNodeForName(parent_node)
        analysis_node = self.create_node()
        analysis_node.setName('{}:{}'.format(parent_node.getName(), self.nametag))
        analysis_node.set_position(lattice.getNodePositionsDict()[parent_node][0])
        analysis_node.setTiltAngle(-parent_node.getTiltAngle())
        parent_node.addChildNode(analysis_node, parent_node.ENTRANCE)
        return analysis_node
    
    def insert_throughout(self, lattice, dense=False, min_sep=1e-5):
        """Insert analysis nodes as child nodes in the lattice.
        
        If `dense`, insert at every part of every node rather than just at every node.
        """
        if not dense:
            return [self.insert_at_entrance(lattice, node) for node in lattice.getNodes()]
        
        analysis_nodes = []
        
        def add_analysis_node_as_child(parent_node, part_index, position):
            analysis_node = self.create_node()
            analysis_node.setName('{}:{}:{}'.format(parent_node.getName(), self.nametag, part_index))
            analysis_node.set_position(position)
            analysis_node.setTiltAngle(-parent_node.getAllChildren()[0].getTiltAngle())
            parent_node.addChildNode(analysis_node, AccNode.BODY, part_index, AccNode.BEFORE)
            analysis_nodes.append(analysis_node)

        for (parent_node, part_index, position) in node_index_position_list(lattice, min_sep):
            add_analysis_node_as_child(parent_node, part_index, position)
            
        return analysis_nodes