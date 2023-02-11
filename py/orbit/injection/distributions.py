"""Injected phase space distributions. This is not a parallel version!"""
import math
import random
import sys

from orbit.utils.consts import speed_of_light


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


class UniformLongDist:
    """Uniform longitudinal distribution."""

    def __init__(self, zmin=None, zmax=None, sp=None, eoffset=None, deltaEfrac=None):
        self.name = "UniformLongDist"
        self.zmin = zmin
        self.zmax = zmax
        self.sp = sp
        self.ekinetic = sp.kinEnergy()
        self.eoffset = eoffset
        self.deltaEfrac = deltaEfrac

    def getCoordinates(self):
        zinj = self.zmin + (self.zmax - self.zmin) * random.random()
        dEinj = self.eoffset + self.ekinetic * self.deltaEfrac * (1.0 - 2.0 * random.random())
        return (zinj, dEinj)


class UniformLongDistPaint:
    """Time-dependent uniform longitudinal distribution with user-defined functions for zmin/zmax."""

    def __init__(
        self, 
        zminFunc=None, 
        zmaxFunc=None, 
        sp=None,
        eoffset=None, 
        deltaEfrac=None,
    ):
        # checks for correct input argument types (to a certain extent)
        if not (type(zminFunc) is list and type(zmaxFunc) is list):
            sys.exit(
                "ERROR from class 'UniformLongDistPaint': input \
            arguments zminFunc and zmaxFunc must be a list of pairs"
            )
        else:
            if len(zminFunc) < 2 or len(zmaxFunc) < 2:
                sys.exit(
                    "ERROR from class 'UniformLongDistPaint': \
                input arguements zminFunc and zmaxFunc must have at \
                least two elements"
                )
            else:
                if not (type(zminFunc[0]) is list and type(zmaxFunc[0]) is list):
                    sys.exit(
                        "ERROR from class 'UniformLongDistPaint': \
                    input arguements zminFunc and zmaxFunc must be a \
                    list of pairs"
                    )

        # sorts functions for quick interpolation calculations
        zminFunc = sorted(zminFunc, key=lambda x: x[0])
        zmaxFunc = sorted(zmaxFunc, key=lambda x: x[0])

        self.name = "UniformLongDistPaint"
        self.zminFunc = zminFunc
        self.zmaxFunc = zmaxFunc
        self.sp = sp
        self.ekinetic = sp.kinEnergy()
        self.eoffset = eoffset
        self.deltaEfrac = deltaEfrac
        # These help vary the number of macro particles according to the PW
        self.frac_change = 1.0
        self.last_length = -1

    def getCoordinates(self):
        # time in seconds
        zminNow = interpolate(self.zminFunc, self.sp.time())
        zmaxNow = interpolate(self.zmaxFunc, self.sp.time())
        if zminNow >= zmaxNow:
            print("Warning from getCoordinates call: zmin >= zmax")
        length = zmaxNow - zminNow
        if self.last_length != -1:
            self.frac_change = length / self.last_length
        self.last_length = length
        zinj = zminNow + (zmaxNow - zminNow) * random.random()
        dEinj = self.eoffset + self.ekinetic * self.deltaEfrac * (1.0 - 2.0 * random.random())
        return (zinj, dEinj)


class GULongDist:
    """
    This class generates random intial longitudinal coordinates for a
    distribution uniform in phi and gaussian in dE.
    """

    def __init__(
        self,
        zmin=None,
        zmax=None,
        sp=None,
        emean=None,
        esigma=None,
        etrunc=None,
        emin=None,
        emax=None,
    ):
        self.name = "GULongDist"
        self.zmin = zmin
        self.zmax = zmax
        self.sp = sp
        self.ekinetic = sp.kinEnergy()
        self.emean = emean
        self.esigma = esigma
        self.etrunc = etrunc
        self.emin = emin
        self.emax = emax

    def getCoordinates(self):
        zinj = self.zmin + (self.zmax - self.zmin) * random.random()
        tol = 1e-6
        ymin = 0.0
        ymax = 10.0
        pmin = 0.0
        pmax = 1.0

        if self.etrunc != 0:
            if self.emin >= self.emean:
                pmin = 0.5 + 0.5 * erf(
                    (self.emin - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmin = 0.5 - 0.5 * erf(
                    (self.emean - self.emin) / (math.sqrt(2.0) * self.esigma)
                )

            if self.emax >= self.emean:
                pmax = 0.5 + 0.5 * erf(
                    (self.emax - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmax = 0.5 - 0.5 * erf(
                    (self.emean - self.emax) / (math.sqrt(2.0) * self.esigma)
                )

        prand = pmin + (pmax - pmin) * random.random()

        while (erf(ymax) - math.fabs(2.0 * prand - 1.0)) < 0.0:
            ymax *= 10.0

        root = rootNorm(ymin, ymax, prand, tol)

        if prand >= 0.5:
            einj = self.emean + math.sqrt(2.0) * self.esigma * root
        else:
            einj = self.emean - math.sqrt(2.0) * self.esigma * root

        dEinj = einj - self.ekinetic

        return (zinj, dEinj)


class SNSESpreadDist:
    """Uniform longitudinal distribution; gaussian energy distribution with
    additional sinusoidal energy spread and random centroid jitter.
    """

    def __init__(
        self,
        lattlength=1.0,
        zmin=None,
        zmax=None,
        tailfraction=0.0,
        sp=None,
        emean=None,
        esigma=None,
        etrunc=None,
        emin=None,
        emax=None,
        ecparams=None,
        esparams=None,
    ):
        self.name = "SNSESpreadDist"
        self.lattlength = lattlength
        self.zmin = zmin
        self.zmax = zmax
        self.tailfraction = tailfraction
        self.sp = sp
        self.emean = emean
        self.ekinetic = sp.kinEnergy()
        self.esigma = esigma
        self.etrunc = etrunc
        self.emin = emin
        self.emax = emax
        self.ecparams = ecparams
        self.esparams = esparams

    def getCoordinates(self):
        if random.random() > self.tailfraction * self.lattlength / (
            self.lattlength - self.zmax + self.zmin
        ):
            # Put it in the main distribution
            zinj = self.zmin + (self.zmax - self.zmin) * random.random()
        else:
            # Put it in an extended tail
            zinj = -self.lattlength / 2.0 + self.lattlength * random.random()

        tol = 1.0e-6
        ymin = 0.0
        ymax = 10.0
        pmin = 0.0
        pmax = 1.0

        if self.etrunc != 0:
            if self.emin >= self.emean:
                pmin = 0.5 + 0.5 * erf(
                    (self.emin - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmin = 0.5 - 0.5 * erf(
                    (self.emean - self.emin) / (math.sqrt(2.0) * self.esigma)
                )

            if self.emax >= self.emean:
                pmax = 0.5 + 0.5 * erf(
                    (self.emax - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmax = 0.5 - 0.5 * erf(
                    (self.emean - self.emax) / (math.sqrt(2.0) * self.esigma)
                )

        prand = pmin + (pmax - pmin) * random.random()

        while (erf(ymax) - math.fabs(2.0 * prand - 1.0)) < 0.0:
            ymax *= 10.0

        root = rootNorm(ymin, ymax, prand, tol)

        if prand >= 0.5:
            einj = self.emean + math.sqrt(2.0) * self.esigma * root
        else:
            einj = self.emean - math.sqrt(2.0) * self.esigma * root

        # If ecmin == 0, use a Gaussian distribution.
        # If ecmin >= 0, then use a truncated Gaussian distribution.
        ecmean = self.ecparams["ecmean"]
        ecsigma = self.ecparams["ecsigma"]
        ectrunc = self.ecparams["ectrunc"]
        ecmin = self.ecparams["ecmin"]
        ecmax = self.ecparams["ecmax"]
        ecdrifti = self.ecparams["ecdrifti"]
        ecdriftf = self.ecparams["ecdriftf"]
        drifttime = self.ecparams["drifttime"]

        pmin = 0.0
        pmax = 1.0
        ec = 0.0

        ecdrift = ecdrifti + (ecdriftf - ecdrifti) * self.sp.time() / drifttime

        if ectrunc != 0:
            if ecmin >= ecmean:
                pmin = 0.5 + 0.5 * erf((ecmin - ecmean) / (math.sqrt(2.0) * ecsigma))
            else:
                pmin = 0.5 - 0.5 * erf((ecmean - ecmin) / (math.sqrt(2.0) * ecsigma))
            if ecmax >= ecmean:
                pmax = 0.5 + 0.5 * erf((ecmax - ecmean) / (math.sqrt(2.0) * ecsigma))
            else:
                pmax = 0.5 + 0.5 * erf((ecmean - ecmax) / (math.sqrt(2.0) * ecsigma))

        prand = pmin + (pmax - pmin) * random.random()

        while (erf(ymax) - math.fabs(2.0 * prand - 1)) < 0.0:
            ymax *= 10.0

        root = rootNorm(ymin, ymax, prand, tol)

        if prand >= 0.5:
            ec = ecmean + ecdrift + math.sqrt(2.0) * ecsigma * root
        else:
            ec = ecmean + ecdrift - math.sqrt(2.0) * ecsigma * root

        esnu = self.esparams['esnu']
        esphase = self.esparams['esphase']
        esmax = self.esparams['esmax']
        nulltime = self.esparams['nulltime']

        iphase = esnu * self.sp.time()
        phasec = 2.0 * math.pi * (esnu * self.sp.time() - iphase)
        turntime = self.lattlength / (self.sp.beta() * speed_of_light)
        phasephifac = esnu * turntime
        phasephi = phasephifac * zinj * math.pi / (self.lattlength / 2.0)
        phase = phasec + phasephi + esphase
        es = esmax * math.sin(phase)
        tfac = 0.0

        if nulltime > 0.0:
            if self.sp.time() > nulltime:
                tfac = 0.0
            else:
                tfac = math.pow(1.0 - self.sp.time() / nulltime, 0.5)

        es *= tfac
        dEinj = einj + ec + es - self.ekinetic
        return (zinj, dEinj)



class SNSESpreadDistPaint:
    """
    This class generates time-dependent SNSESpreadDistPaint distribution
    coordinates according to user-defined (mathematical) functions for
    zmin and zmax
    """

    def __init__(
        self,
        lattlength=None,
        zminFunc=None,
        zmaxFunc=None,
        tailfraction=None,
        sp=None,
        emean=None,
        esigma=None,
        etrunc=None,
        emin=None,
        emax=None,
        ecparams=None,
        esparams=None,
    ):
        # checks for correct input arguement types (to a certain extent)
        if not (type(zminFunc) is list and type(zmaxFunc) is list):
            raise ValueError('zminFunc and zmaxFunc must be list of pairs.')
        else:
            if len(zminFunc) < 2 or len(zmaxFunc) < 2:
                raise ValueError('zminFunc and zmaxFunc must have at least two elements')
            else:
                if not (type(zminFunc[0]) is list and type(zmaxFunc[0]) is list):
                    raise ValueError('zminFunc and zmaxFunc must be list of pairs.')

        # sort functions for quick interpolation calculations
        zminFunc = sorted(zminFunc, key=lambda x: x[0])
        zmaxFunc = sorted(zmaxFunc, key=lambda x: x[0])

        self.name = "SNSESpreadDistPaint"
        self.lattlength = lattlength
        self.zminFunc = zminFunc
        self.zmaxFunc = zmaxFunc
        self.tailfraction = tailfraction
        self.sp = sp
        self.emean = emean
        self.ekinetic = sp.kinEnergy()
        self.esigma = esigma
        self.etrunc = etrunc
        self.emin = emin
        self.emax = emax
        self.ecparams = ecparams
        self.esparams = esparams
        # These help vary the number of macro partices according to PW
        self.frac_change = 1.0
        self.last_length = -1

    def getCoordinates(self):
        zminNow = interpolate(self.zminFunc, self.sp.time())
        zmaxNow = interpolate(self.zmaxFunc, self.sp.time())
        length = zmaxNow - zminNow

        if self.last_length != -1:
            self.frac_change = length / self.last_length
        self.last_length = length

        if random.random() > self.tailfraction * self.lattlength / (
            self.lattlength - zmaxNow + zminNow
        ):
            # Put it in the main distribution
            zinj = zminNow + (zmaxNow - zminNow) * random.random()
        else:
            # Put it in an extended tail
            zinj = -self.lattlength / 2.0 + self.lattlength * random.random()

        tol = 1e-6
        ymin = 0.0
        ymax = 10.0
        pmin = 0.0
        pmax = 1.0

        if self.etrunc != 0:
            if self.emin >= self.emean:
                pmin = 0.5 + 0.5 * erf(
                    (self.emin - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmin = 0.5 - 0.5 * erf(
                    (self.emean - self.emin) / (math.sqrt(2.0) * self.esigma)
                )

            if self.emax >= self.emean:
                pmax = 0.5 + 0.5 * erf(
                    (self.emax - self.emean) / (math.sqrt(2.0) * self.esigma)
                )
            else:
                pmax = 0.5 - 0.5 * erf(
                    (self.emean - self.emax) / (math.sqrt(2.0) * self.esigma)
                )

        prand = pmin + (pmax - pmin) * random.random()

        while (erf(ymax) - math.fabs(2.0 * prand - 1)) < 0.0:
            ymax *= 10.0

        root = rootNorm(ymin, ymax, prand, tol)

        if prand >= 0.5:
            einj = self.emean + math.sqrt(2.0) * self.esigma * root
        else:
            einj = self.emean - math.sqrt(2.0) * self.esigma * root

        # If ecmin = 0  then do regular Gaussian distribution.
        # If ecmin >= 0, then this will indicate that a truncated \
        # Gaussian distribution is desired.

        (
            ecmean,
            ecsigma,
            ectrunc,
            ecmin,
            ecmax,
            ecdrifti,
            ecdriftf,
            drifttime,
        ) = self.ecparams

        pmin = 0.0
        pmax = 1.0
        ec = 0.0

        ecdrift = ecdrifti + (ecdriftf - ecdrifti) * self.sp.time() / drifttime

        if ectrunc != 0:
            if ecmin >= ecmean:
                pmin = 0.5 + 0.5 * erf((ecmin - ecmean) / (math.sqrt(2.0) * ecsigma))
            else:
                pmin = 0.5 - 0.5 * erf((ecmean - ecmin) / (math.sqrt(2.0) * ecsigma))
            if ecmax >= ecmean:
                pmax = 0.5 + 0.5 * erf((ecmax - ecmean) / (math.sqrt(2.0) * ecsigma))
            else:
                pmax = 0.5 + 0.5 * erf((ecmean - ecmax) / (math.sqrt(2.0) * ecsigma))

        prand = pmin + (pmax - pmin) * random.random()

        while (erf(ymax) - math.fabs(2.0 * prand - 1)) < 0.0:
            ymax *= 10.0

        root = rootNorm(ymin, ymax, prand, tol)

        if prand >= 0.5:
            ec = ecmean + ecdrift + math.sqrt(2.0) * ecsigma * root
        else:
            ec = ecmean + ecdrift - math.sqrt(2.0) * ecsigma * root

        (esnu, esphase, esmax, nulltime) = self.esparams

        iphase = esnu * self.sp.time()
        phasec = 2.0 * math.pi * (esnu * self.sp.time() - iphase)
        turntime = self.lattlength / (self.sp.beta() * speed_of_light)
        phasephifac = esnu * turntime
        phasephi = phasephifac * zinj * math.pi / (self.lattlength / 2.0)
        phase = phasec + phasephi + esphase
        es = esmax * math.sin(phase)
        tfac = 0.0

        if nulltime > 0.0:
            if self.sp.time() > nulltime:
                tfac = 0.0
            else:
                tfac = math.pow(1.0 - self.sp.time() / nulltime, 0.5)

        es *= tfac

        dEinj = einj + ec + es - self.ekinetic

        return (zinj, dEinj)



class ArbitraryLongDist:
    """User-supplied phase (z) and energy distribution arrays"""

    def __init__(self, phaselength=None, phase=None, phaseProb=None, dE=None, dEProb=None):
        self.name = "ArbitraryLongDist"
        self.phaselength = phaselength
        self.phase = phase
        self.phaseProb = phaseProb
        self.dE = dE
        self.dEProb = dEProb
        self.z = self.phasetoz(self.phaselength, self.phase)
        self.phaseProbInt = self.getProbInt(phaseProb)
        self.dEProbInt = self.getProbInt(dEProb)

    def reset(self, phaselength=None, phase=None, phaseProb=None, dE=None, dEProb=None):
        self.phaselength = phaselength
        self.phase = phase
        self.phaseProb = phaseProb
        self.dE = dE
        self.dEProb = dEProb
        self.z = self.phasetoz(self.phaselength, self.phase)
        self.phaseProbInt = self.getProbInt(phaseProb)
        self.dEProbInt = self.getProbInt(dEProb)

    def phasetoz(self, phaselength, phase):
        z = []
        coeff = -phaselength / (2.0 * math.pi)
        for i in range(len(phase)):
            z.append(coeff * phase[i])
        return z

    def getProbInt(self, Prob):
        ProbInt = [0.0]
        ProbTot = 0.0
        for i in range(1, len(Prob)):
            ProbTot = ProbTot + 0.5 * (Prob[i - 1] + Prob[i])
            ProbInt.append(ProbTot)
        for i in range(len(Prob)):
            ProbInt[i] = ProbInt[i] / ProbTot
        return ProbInt

    def getCoord(self, x, ProbInt):
        y = random.random()
        istop = 0
        imin = len(ProbInt) - 2
        imax = len(ProbInt) - 1
        for i in range(len(ProbInt)):
            if y < ProbInt[i]:
                imin = i - 1
                imax = i
                istop = 1
            if istop == 1:
                break
        frac = (y - ProbInt[imin]) / (ProbInt[imax] - ProbInt[imin])
        coord = x[imin] + frac * (x[imax] - x[imin])
        return coord

    def getCoordinates(self):
        zinj = self.getCoord(self.z, self.phaseProbInt)
        dEinj = self.getCoord(self.dE, self.dEProbInt)
        return (zinj, dEinj)


def interpolate(List, time):
    ListMax = List[len(List) - 1][0]
    ListMin = List[0][0]

    # boundry cases
    if time >= ListMax:
        p1 = List[len(List) - 2]
        p2 = List[len(List) - 1]
    elif time <= ListMin:
        p1 = List[1]
        p2 = List[0]
    else:
        # binary search that returns the closest x-coordinate
        e = len(List) - 1  # end
        f = 0  # front
        i = int(math.floor((e + f) / 2))
        closest = i
        while f <= e:
            if math.fabs(time - List[i][0]) <= math.fabs(time - List[closest][0]):
                closest = i
            if List[i][0] > time:
                e = i - 1
                i = int(math.floor((e + f) / 2))
            elif List[i][0] < time:
                f = i + 1
                i = int(math.floor((e + f) / 2))
            else:
                return List[i][1]

        if List[i][0] < time:
            p1 = List[i]
            p2 = List[i + 1]
        else:
            p1 = List[i - 1]
            p2 = List[i]

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]

    return m * time + b


def rootNorm(ymin, ymax, prand, tol):
    """Finds the roots of the (Gauss) function."""
    imax = 40
    rtbis = 0.0
    dx = 0.0
    xmid = 0.0
    fmid = 0.0

    if (erf(ymin) - math.fabs(2.0 * prand - 1.0)) < 0.0:
        rtbis = ymin
        dx = ymax - ymin
    else:
        rtbis = ymax
        dx = ymin - ymax

    for i in xrange(imax):
        dx = dx * 0.5
        xmid = rtbis + dx
        fmid = erf(xmid) - math.fabs(2.0 * prand - 1.0)
        if fmid <= 0.0:
            rtbis = xmid
        if (math.fabs(dx) < tol) or (fmid == 0):
            return rtbis

    return rtbis


def erf(z):
    t = 1.0 / (1.0 + 0.5 * abs(z))
    # use Horner's method
    ans = 1 - t * math.exp(-z*z - 1.26551223 + \
                               t * (1.00002368 + \
                                    t * (0.37409196 + \
                                         t * (0.09678418 + \
                                              t * (-0.18628806 + \
                                                   t * (0.27886807 + \
                                                        t * (-1.13520398 + \
                                                             t * (1.48851587 + \
                                                                  t * (-0.82215223 + t * (0.17087277))))))))))

    if z >= 0.0:
        return ans
    else:
        return -ans