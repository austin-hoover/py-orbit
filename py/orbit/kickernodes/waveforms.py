#!/usr/bin/env python

"""
This is not a parallel version!
"""

import math
import random
import sys

class rootTWaveform:
	"""
	This class has sqrt(time) waveform.
	"""
	def __init__(self, syncpart, lattlength, duration, startamp, endamp):
		self.name = "RootTWaveform"
		self.syncpart = syncpart
		self.lattlength = lattlength
		self.duration = duration
		self.startamp = startamp
		self.endamp = endamp

	def getStrength(self):
		time = self.syncpart.time()
		amp = self.startamp - self.endamp;
		factor = amp * (1.0 - math.sqrt(time/self.duration)) \
                         + self.endamp
                if(time > self.duration):
                        factor = self.endamp
		return factor

class flatTopWaveform:
	""" 
	This class has flat top (const) waveform.
	"""
	def __init__(self, amp):
		self.name = "FlatTopWaveform"
		self.amp = amp

	def getStrength(self):
		return self.amp
