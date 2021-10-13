#!/usr/bin/env python

"""
This is not a parallel version! 
"""
import math
import random
import sys
from bunch import Bunch
from orbit.teapot import TEAPOT_MATRIX_Lattice


class simpleBump:
    """This routine adds a transverse particle coordinate bump."""
    def __init__(self, bunch, xbump, xpbump, ybump, ypbump, waveform=None):
        self.bunch = bunch
        self.xbump = xbump
        self.ybump = ybump
        self.xpbump = xpbump
        self.ypbump = ypbump
        self.waveform = waveform

    def bump(self):
        strength = 1.0
        if self.waveform:
            strength = self.waveform.getAmplitude()
        nparts = self.bunch.getSize()
        for i in range(nparts):
            newx = self.bunch.x(i) + self.xbump * strength
            self.bunch.x(i, newx)
            newxp = self.bunch.xp(i) + self.xpbump * strength
            self.bunch.xp(i, newxp)
            newy = self.bunch.y(i) + self.ybump * strength
            self.bunch.y(i, newy)
            newyp = self.bunch.yp(i) + self.ypbump * strength
            self.bunch.yp(i, newyp)

    def getLength(self):
        return 0.0

    def setWaveform(self, waveform):
        self.waveform = waveform