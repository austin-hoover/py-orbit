"""Generate JOHO phase space distributions. This is not a parallel version."""
import math
import random
import sys


class JohoTransverse:
    """Transverse Joho distribution generator."""

    def __init__(
        self,
        order=1,
        alpha=0.0,
        beta=1.0,
        emitlim=1.0,
        centerpos=0.0,
        centermom=0.0,
        tailfraction=0.0,
        tailfactor=1.0,
    ):
        self.name = "JohoTransverse"
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.emitlim = emitlim
        self.centerpos = centerpos
        self.centermom = centermom
        self.tailfraction = tailfraction
        self.tailfactor = tailfactor
        self.__initialize()

    def __initialize(self):
        self.emit = self.emitlim * 2.0 / (1.0 + self.order)
        self.pos = math.sqrt(self.emit)
        self.gamma = (1.0 + self.alpha * self.alpha) / self.beta
        self.mom = math.sqrt(self.emit * self.gamma)
        self.coschi = math.sqrt(1.0 / (1.0 + (self.alpha * self.alpha)))
        self.sinchi = -self.alpha * self.coschi
        self.poslength = math.sqrt(self.emitlim * self.beta)
        self.momlength = math.sqrt(self.emitlim * self.gamma)
        self.orderinv = 1.0 / self.order
        self.emitrms = 0.5 * self.emitlim / (1.0 + self.order)

    def getCoordinates(self):
        s1 = random.random()
        s2 = random.random()
        a = math.sqrt(1 - pow(s1, self.orderinv))
        al = 2.0 * math.pi * s2
        u = a * math.cos(al)
        v = a * math.sin(al)
        dpos = self.poslength * u
        dmom = self.momlength * (u * self.sinchi + v * self.coschi)
        if self.tailfraction > 0.0:
            if random.random() < self.tailfraction:
                dpos *= self.tailfactor
                dmom *= self.tailfactor
        pos = self.centerpos + dpos
        mom = self.centermom + dmom
        return (pos, mom)


class JohoLongitudinal:
    """Longitudinal Joho distribution generator."""

    def __init__(
        self,
        order=1,
        zlim=1.0,
        dElim=1.0,
        nlongbunches=0,
        deltazbunch=0.0,
        deltaznotch=0.0,
        tailfraction=0.0,
        tailfactor=1.0,
    ):
        self.name = "JohoLongitudinal"
        self.order = order
        self.zlim = zlim
        self.dElim = dElim
        self.nlongbunches = nlongbunches
        self.deltazbunch = deltazbunch
        self.deltaznotch = deltaznotch
        self.tailfraction = tailfraction
        self.tailfactor = tailfactor

    def getCoordinates(self):
        orderinv = 1.0 / self.order
        s1 = random.random()
        s2 = random.random()
        a = math.sqrt(1.0 - pow(s1, orderinv))
        al = 2.0 * math.pi * s2
        u = a * math.cos(al)
        v = a * math.sin(al)
        zinj = self.zlim * u
        dEinj = self.dElim * v
        factor = 360 / 248.0
        if self.tailfraction > 0.0:
            if random.random() < self.tailfraction:
                zinj *= self.ltailfraction
                dEinj *= self.ltailfraction

        if self.nlongbunches > 1:
            ibunch = int(1 + self.nlongbunches * random.random())
            if self.nlongbunches < ibunch:
                ibunch = self.nlongbunches

            offset = (2.0 * ibunch - self.nlongbunches - 1) / 2.0
            ztemp = offset * self.deltazbunch

            if self.deltaznotch != 0.0:
                while (ztemp < self.deltaznotch / 2.0) & (
                    ztemp > -self.deltaznotch / 2.0
                ):
                    ibunch = int(1 + self.nlongbunches * random.random())
                    if self.nlongbunches < ibunch:
                        ibunch = self.nlongbunches
                    offset = (2.0 * ibunch - self.nlongbunches - 1) / 2.0
                    ztemp = offset * self.deltazbunch
            zinj += ztemp

        return (zinj, dEinj)
