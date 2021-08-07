import sys
import os

from orbit.utils   import orbitFinalize
from orbit.utils   import NamedObject
from orbit.utils   import TypedObject

from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode

import orbit

class AccLattice(NamedObject, TypedObject):
    """
    Class. The accelerator lattice class contains child nodes.
    """

    ENTRANCE = AccActionsContainer.ENTRANCE
    BODY     = AccActionsContainer.BODY
    EXIT     = AccActionsContainer.EXIT

    BEFORE   = AccActionsContainer.BEFORE
    AFTER    = AccActionsContainer.AFTER

    def __init__(self, name = "no name"):
        """
        Constructor. Creates an empty accelerator lattice.
        """
        NamedObject.__init__(self, name)
        TypedObject.__init__(self, "lattice")
        self.__length = 0.
        self.__isInitialized = False
        self.__children = []
        self.__childPositions = {}

    def initialize(self):
        """
        Method. Initializes the lattice and child node structures.
        """
        res_dict = {}
        for node in self.__children:
            if(res_dict.has_key(node)):
                msg = "The AccLattice class instance should not have duplicate nodes!"
                msg = msg + os.linesep
                msg = msg + "Method initialize():"
                msg = msg + os.linesep
                msg = msg + "Name of node=" + node.getName()
                msg = msg + os.linesep
                msg = msg + "Type of node=" + node.getType()
                msg = msg + os.linesep
                orbitFinalize(msg)
            else:
                res_dict[node] = None
            node.initialize()
        del res_dict

        paramsDict = {}
        actions = AccActionsContainer()
        d = [0.]
        posn = {}

        def accNodeExitAction(paramsDict):
            """
            Nonbound function. Sets lattice length and node
            positions. This is a closure (well, maybe not
            exactly). It uses external objects.
            """
            node = paramsDict["node"]
            parentNode = paramsDict["parentNode"]
            if(isinstance(parentNode, AccLattice)):
                posBefore = d[0]
                d[0] += node.getLength()
                posAfter = d[0]
                posn[node]=(posBefore, posAfter)

        actions.addAction(accNodeExitAction, AccNode.EXIT)
        self.trackActions(actions, paramsDict)
        self.__length = d[0]
        self.__childPositions = posn
        self.__isInitialized = True

    def isInitialized(self):
        """
        Method. Returns the initialization status (True or False).
        """
        return self.__isInitialized

    def addNode(self, node, index = -1):
        """
        Method. Adds a child node into the lattice. If the user
        specifies the index >= 0 the element will be inserted in
        the specified position into the children array
        """
        if(isinstance(node, AccNode) == True): 
            if(index < 0): 
                self.__children.append(node)
            else:
                self.__children.insert(index,node)
            self.__isInitialized = False

    def getNodes(self):
        """
        Method. Returns a list of all children
        of the first level in the lattice.
        """
        return self.__children

    def setNodes(self,childrenNodes):
        """
        Method. Set up a new list of all children
        of the first level in the lattice.
        """	
        self.__children	 = childrenNodes

    def getNodeForName(self,name):
        """
        Method. Returns the node with certain name.
        """
        nodes = []
        for node in self.__children:
            if(node.getName().find(name) == 0):
                nodes.append(node)
        if(len(nodes) == 1):
            return nodes[0]
        else:
            if(len(nodes) == 0):
                return None
            else:
                msg = "The AccLattice class. Method getNodeForName found many nodes instead of one!"
                msg = msg + os.linesep
                msg = msg + "looking for name="+name
                msg = msg + os.linesep
                msg = msg + "found nodes:"
                for node in nodes:
                    msg = msg + " " + node.getName()
                msg = msg + os.linesep
                msg = msg + "Please use getNodesForName method instead."
                msg = msg + os.linesep				
                orbitFinalize(msg)
        
    def getNodesForNames(self, names):
        """Return list of nodes from list of names."""
        nodes = []
        for node in self.__children:
            if node.getName() in names:
                nodes.append(node)
        return nodes

    def getNodesOfClass(self,class_of_node):
        """
        Method. Returns nodes off a certain class.
        """
        nodes = []
        for node in self.__children:
            if(isinstance(node,class_of_node)):
                nodes.append(node)
        return nodes		

    def getNodesForSubstring(self,sub, no_sub = None):
        """
        Method. Returns nodes with names each of them has the certain substring. 
        It is also possible to specify the unwanted substring as no_sub parameter.
        """
        nodes = []
        for node in self.__children:
            if(no_sub == None):
                if(node.getName().find(sub) >= 0):
                    nodes.append(node)
            else:
                if(node.getName().find(sub) >= 0 and node.getName().find(no_sub) < 0):
                    nodes.append(node)		
        return nodes		

    def getNodeIndex(self,node):
        """
        Method. Returns the index of the node in the upper level of the lattice children-nodes.
        """		
        return self.__children.index(node)

    def getNodePositionsDict(self):
        """
        Method. Returns a dictionary of
        {node:(start position, stop position)}
        tuples for all children of the first level in the lattice.
        """
        return self.__childPositions

    def getLength(self):
        """
        Method. Returns the physical length of the lattice.
        """
        return self.__length

    def reverseOrder(self):
        """
        This method is used for a lattice reversal and a bunch backtracking.
        This method will reverse the order of the children nodes. It will 
        apply the reverse recursively to the all children nodes.
        """
        self.__children.reverse()
        for node in self.__children:
            node.reverseOrder()
        self.initialize()

    def structureToText(self):
        """
        Returns the text with the lattice structure.
        """
        txt = "==== START ==== Lattice =" + self.getName() + "  L=" + str(self.getLength()) 
        txt += os.linesep
        for node in self.__children:
            txt += node.structureToText("")
        txt += "==== STOP  ==== Lattice =" + self.getName() + "  L=" + str(self.getLength())
        txt += os.linesep
        return txt

    def _getSubLattice(self, accLatticeNew, index_start = -1, index_stop = -1):
        """
        It returns the sub-accelerator lattice with children with
        indexes between index_start and index_stop, inclusive. The
        subclasses of AccLattice should NOT override this method.
        """
        if(index_start < 0): index_start = 0
        if(index_stop < 0): index_stop = len(self.__children) - 1 
        #clear the node array in the new sublattice
        accLatticeNew.setNodes([])
        for node in self.__children[index_start:index_stop+1]:
            accLatticeNew.addNode(node)
        accLatticeNew.initialize()
        return accLatticeNew

    def getSubLattice(self, index_start = -1, index_stop = -1,):
        """
        It returns the sub-accelerator lattice with children with
        indexes between index_start and index_stop inclusive. The
        subclasses of AccLattice should override this method to replace
        AccLattice() constructor by the sub-class type constructor
        """
        return self._getSubLattice( AccLattice(),index_start,index_stop)
    
    def split(self, max_node_length):
        """Split nodes into parts so no part is longer than max_node_length."""
        for node in self.getNodes():
            node_length = node.getLength()
            if node_length > max_node_length:
                node.setnParts(1 + int(node_length / max_node_length))
        
    def set_fringe(self, switch):
        """Turn on(off) fringe field calculation for all nodes."""
        for node in self.getNodes():
            node.setUsageFringeFieldIN(switch)
            node.setUsageFringeFieldOUT(switch)

    def trackActions(self, actionsContainer, paramsDict = {}, index_start = -1, index_stop = -1):
        """
        Method. Tracks the actions through all nodes in the lattice. The indexes are inclusive.
        """
        paramsDict["lattice"] = self
        paramsDict["actions"] = actionsContainer
        if(not paramsDict.has_key("path_length")): paramsDict["path_length"] = 0.
        if(index_start < 0): index_start = 0
        if(index_stop < 0): index_stop = len(self.__children) - 1 		
        for node in self.__children[index_start:index_stop+1]:
            paramsDict["node"] = node
            paramsDict["parentNode"] = self
            node.trackActions(actionsContainer, paramsDict)
