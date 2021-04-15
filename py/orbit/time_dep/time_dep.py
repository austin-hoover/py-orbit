"""
Module. Includes classes for all time-dependent lattice.
"""
import sys
import os
import math

from orbit.teapot import TEAPOT_Lattice
from orbit.parsers.mad_parser import MAD_Parser, MAD_LattLine
from orbit.lattice import AccNode, AccActionsContainer
from orbit.time_dep import waveform


class TIME_DEP_Lattice(TEAPOT_Lattice):
    """The subclass of the TEAPOT_Lattice.
    
    A TIME_DEP_Lattice has the ability to set time-dependent parameters to a
    Lattice. Multi-turn tracking is also available.
    """
    def __init__(self, name="no name"):
        TEAPOT_Lattice.__init__(self, name)
        self.__latticeDict = {}
        self.__TDNodeDict = {}
        self.__turns = 1

    def setLatticeOrder(self):
        """Set the time-dependent lattice names to the lattice."""
        accNodes = self.getNodes()
        elemInLine = {}
        for i in range(len(accNodes)):
            elem = accNodes[i]
            elemname = elem.getName()
            if elemInLine.has_key(elemname):
                elemInLine[elemname] += 1
            else:
                elemInLine[elemname] = 1
            node = self.getNodes()[i]
            param_name = node.getName() + "_" + str(elemInLine[elemname])
            node.setParam("TPName", param_name)
            #node.setParam("sequence", i+1)
            #print "debug node", node.getName(), \
                                #node.getParamsDict()

    def setTimeDepNode(self, TPName, waveform):
        """Set the waveform function to the TP node before track."""
        flag = 0
        for node in self.getNodes():
                if TPName == node.getParam("TPName"):
                    flag = 1
                    node.setWaveform(waveform)
                    self.__TDNodeDict[TPName] = node
        if not flag:
            print "{} is not found.".format(TPName)
            sys.exit(1)

    def trackBunchTurns(self, bunch, paramsDict):
        """Tracks the bunch through the lattice with multi-turn."""
        turns = self.__turns
        #start
        for i in range(turns - 1):
            self.trackBunch(bunch, paramsDict)
            syncPart = bunch.getSyncParticle()
        #getsublattice
        #sublattice.trackBunch(bunch)

    def setTurns(self, turns, startPosition = 0, endPosition = -1):
        """
        Sets the turns and start end position before track.
        """
        startNode = StartNode("start node")
        endNode = EndNode("end node")
        self.addNode(startNode, startPosition)
        self.addNode(endNode, endPosition)
        self.__turns = turns
        #print self.getNodes()


class StartNode(AccNode):
    
    def __init__(self, name = "no name"):
        AccNode.__init__(self, name)
        self.setType("start node")

    def track(self, paramsDict):
        bunch = paramsDict["bunch"]
        #bunch.getSyncParticle().time(0.)


class EndNode(AccNode):
    
    def __init__(self, name = "no name"):
        AccNode.__init__(self,name)
        self.setType("end node")

    def track(self, paramsDict):
        pass
