import sys
import os
import math

class ConstantWaveform:
	"""
	Waveform of constant strength.
	"""
	def __init__(self, syncpart, lattlength, strength):
		self.name = "constant waveform"
		self.syncpart = syncpart
                self.lattlength = lattlength
                self.strength = strength

	def getStrength(self):
		return self.strength


class SquareRootWaveform:
	"""
	Square root waveform.
	"""
	def __init__(self, syncpart, lattlength, t1, t2, t1amp, t2amp):
		self.name = "square root waveform"
		self.syncpart = syncpart
		self.lattlength = lattlength
                self.t1 = t1
                self.t2 = t2
                self.t1amp = t1amp
                self.t2amp = t2amp

	def getStrength(self):
		time = self.syncpart.time()
                if(time < self.t1):
                        strength = self.t1amp
                elif(time > self.t2):
                        strength = self.t2amp
                else:
                        dt = math.sqrt((time - self.t1) / (self.t2 - self.t1))
                        strength = ((self.t1amp - self.t2amp) * (1.0 - dt) \
                                         + self.t2amp)
		return strength


class LinearWaveform:
	"""
	Linear strength variation between t1 and t2
	"""
	def __init__(self, syncpart, lattlength, t1, t2, t1amp, t2amp):
		self.name = "linear waveform"
		self.syncpart = syncpart
		self.lattlength = lattlength
                self.t1 = t1
                self.t2 = t2
                self.t1amp = t1amp
                self.t2amp = t2amp

	def getStrength(self):
		time = self.syncpart.time()
                if(time < self.t1):
                        strength = self.t1amp
                elif(time > self.t2):
                        strength = self.t2amp
		else:
                        dt = (time - self.t1) / (self.t2 - self.t1)
                        strength = (self.t2amp - self.t1amp) * dt + self.t1amp
		return strength
