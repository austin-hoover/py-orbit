"""
This module is a foil node class for TEAPOT lattice
"""

import os
import math

# import the auxiliary classes
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject

# import general accelerator elements and lattice
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker

# import teapot drift class
from orbit.teapot import DriftTEAPOT

# import bump class
from orbit.bumps import simpleBump


class TeapotSimpleBumpNode(DriftTEAPOT):
    """ 
    The kicker node class for TEAPOT lattice
    """
    def __init__(self, bunch, xbump, xpbump, ybump, ypbump, waveform=None, name="bump"):
        """Constructor. Creates the Bump TEAPOT element."""
        DriftTEAPOT.__init__(self, name)
        self.simplebump = simpleBump(bunch, xbump, xpbump, ybump, ypbump, waveform);
        self.setType("Bump")
        self.setLength(0.0)

    def track(self, paramsDict):
        """The simplebump-teapot class implementation of the AccNodeBunchTracker class track(probe) method."""
        length = self.getLength(self.getActivePartIndex())
        bunch = paramsDict["bunch"]
        self.simplebump.bump()

