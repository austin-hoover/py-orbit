import math


class ConstantWaveform:
    
    def __init__(self, amp):
        self.name = "constant waveform"
        self.amp = amp

    def getAmplitude(self):
        return self.amp


class LinearWaveform:
    
    def __init__(self, syncpart, lat_length, t1, t2, amp1, amp2):
        self.name = 'linear waveform'
        self.syncpart = syncpart
        self.t1, self.t2, self.amp1, self.amp2 = t1, t2, amp1, amp2

    def getAmpltidue(self):
        t = self.syncpart.time()
        if t < self.t1:
            amp = self.amp1
        elif t > self.tf:
            amp = self.amp2
        else:
            dt = (time - self.t1) / (self.t2 - self.t1)
            amp = (self.amp2 - self.amp1) * dt + self.amp1
        return amp
    

class SquareRootWaveform:
    
    def __init__(self, syncpart, t1, t2, amp1, amp2):
        self.name = 'square root waveform'
        self.syncpart = syncpart
        self.t1, self.t2, self.amp1, self.amp2 = t1, t2, amp1, amp2

    def getAmpltidue(self):
        t = self.syncpart.time()
        if t < self.t1:
            amp = self.amp1
        elif t > self.tf:
            amp = self.amp2
        else:
            dt = math.sqrt((t - self.t1) / (self.t2 - self.t1))
            amp = ((self.amp1 - self.amp2) * (1.0 - dt) + self.amp2)
        return amp