"""This is not a parallel version!"""
from __future__ import print_function
import collections
import math
import os
import random
import sys

import numpy as np

from bunch import BunchTwissAnalysis
from bunch import BunchTuneAnalysis
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
from orbit.utils import NamedObject
from orbit.utils import orbitFinalize
from orbit.utils import ParamsDictObject
from orbit.utils.consts import speed_of_light
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op


class StatLats:
    """This class gathers delivers the statistical twiss parameters."""

    def __init__(self, filename):
        self.file_out = open(filename, "a")
        self.bunchtwissanalysis = BunchTwissAnalysis()

    def writeStatLats(self, s, bunch, lattlength=0):
        self.bunchtwissanalysis.analyzeBunch(bunch)
        emitx = self.bunchtwissanalysis.getEmittance(0)
        betax = self.bunchtwissanalysis.getBeta(0)
        alphax = self.bunchtwissanalysis.getAlpha(0)
        betay = self.bunchtwissanalysis.getBeta(1)
        alphay = self.bunchtwissanalysis.getAlpha(1)
        emity = self.bunchtwissanalysis.getEmittance(1)
        dispersionx = self.bunchtwissanalysis.getDispersion(0)
        ddispersionx = self.bunchtwissanalysis.getDispersionDerivative(0)
        dispersiony = self.bunchtwissanalysis.getDispersion(1)
        ddispersiony = self.bunchtwissanalysis.getDispersionDerivative(1)

        sp = bunch.getSyncParticle()
        time = sp.time()
        if lattlength > 0:
            time = sp.time() / (lattlength / (sp.beta() * speed_of_light))

        # if mpi operations are enabled, this section of code will
        # determine the rank of the present node
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)

        # only the primary node needs to output the calculated information
        if rank == 0:
            self.file_out.write(
                str(s)
                + "\t"
                + str(time)
                + "\t"
                + str(emitx)
                + "\t"
                + str(emity)
                + "\t"
                + str(betax)
                + "\t"
                + str(betay)
                + "\t"
                + str(alphax)
                + "\t"
                + str(alphay)
                + "\t"
                + str(dispersionx)
                + "\t"
                + str(ddispersionx)
                + "\n"
            )

    def closeStatLats(self):
        self.file_out.close()


class StatLatsSetMember:
    """This class delivers the statistical Twiss parameters."""

    def __init__(self, file):
        self.file_out = file
        self.bunchtwissanalysis = BunchTwissAnalysis()

    def writeStatLats(self, s, bunch, lattlength=0):

        self.bunchtwissanalysis.analyzeBunch(bunch)
        emitx = self.bunchtwissanalysis.getEmittance(0)
        betax = self.bunchtwissanalysis.getBeta(0)
        alphax = self.bunchtwissanalysis.getAlpha(0)
        betay = self.bunchtwissanalysis.getBeta(1)
        alphay = self.bunchtwissanalysis.getAlpha(1)
        emity = self.bunchtwissanalysis.getEmittance(1)
        dispersionx = self.bunchtwissanalysis.getDispersion(0)
        ddispersionx = self.bunchtwissanalysis.getDispersionDerivative(0)
        # dispersiony = self.bunchtwissanalysis.getDispersion(1, bunch)
        # ddispersiony = self.bunchtwissanalysis.getDispersionDerivative(1, bunch)

        sp = bunch.getSyncParticle()
        time = sp.time()

        if lattlength > 0:
            time = sp.time() / (lattlength / (sp.beta() * speed_of_light))

        # if mpi operations are enabled, this section of code will
        # determine the rank of the present node
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)

        # only the primary node needs to output the calculated information
        if rank == 0:
            self.file_out.write(
                str(s)
                + "\t"
                + str(time)
                + "\t"
                + str(emitx)
                + "\t"
                + str(emity)
                + "\t"
                + str(betax)
                + "\t"
                + str(betay)
                + "\t"
                + str(alphax)
                + "\t"
                + str(alphay)
                + "\t"
                + str(dispersionx)
                + "\t"
                + str(ddispersionx)
                + "\n"
            )

    def closeStatLats(self):
        self.file_out.close()

    def resetFile(self, file):
        self.file_out = file


class Moments:
    """This class delivers the beam moments."""

    def __init__(self, filename, order, nodispersion, emitnorm):
        self.file_out = open(filename, "a")
        self.bunchtwissanalysis = BunchTwissAnalysis()
        self.order = order
        if nodispersion == False:
            self.dispterm = -1
        else:
            self.dispterm = 1

        if emitnorm == True:
            self.emitnormterm = 1
        else:
            self.emitnormterm = -1

    def writeMoments(self, s, bunch, lattlength=0):

        sp = bunch.getSyncParticle()
        time = sp.time()
        if lattlength > 0:
            time = sp.time() / (lattlength / (sp.beta() * speed_of_light))

        self.bunchtwissanalysis.computeBunchMoments(
            bunch, self.order, self.dispterm, self.emitnormterm
        )

        # if mpi operations are enabled, this section of code will
        # determine the rank of the present node
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)

        # only the primary node needs to output the calculated information
        if rank == 0:
            self.file_out.write(str(s) + "\t" + str(time) + "\t")
            for i in range(0, self.order + 1):
                for j in range(0, i + 1):
                    self.file_out.write(
                        str(self.bunchtwissanalysis.getBunchMoment(i - j, j)) + "\t"
                    )
            self.file_out.write("\n")

    def closeMoments(self):
        self.file_out.close()


class MomentsSetMember:
    """This class delivers the beam moments."""

    def __init__(self, file, order, nodispersion, emitnorm):
        self.file_out = file
        self.order = order
        self.bunchtwissanalysis = BunchTwissAnalysis()
        if nodispersion == False:
            self.dispterm = -1
        else:
            self.dispterm = 1

        if emitnorm == True:
            self.emitnormterm = 1
        else:
            self.emitnormterm = -1

    def writeMoments(self, s, bunch, lattlength=0):

        sp = bunch.getSyncParticle()
        time = sp.time()

        if lattlength > 0:
            time = sp.time() / (lattlength / (sp.beta() * speed_of_light))

        self.bunchtwissanalysis.computeBunchMoments(
            bunch, self.order, self.dispterm, self.emitnormterm
        )

        # if mpi operations are enabled, this section of code will
        # determine the rank of the present node
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)

        # only the primary node needs to output the calculated information
        if rank == 0:
            self.file_out.write(str(s) + "\t" + str(time) + "\t")
            for i in range(0, self.order + 1):
                for j in range(0, i + 1):
                    self.file_out.write(
                        str(self.bunchtwissanalysis.getBunchMoment(i - j, j)) + "\t"
                    )
            self.file_out.write("\n")

    def resetFile(self, file):
        self.file_out = file


class BPMSignal:
    """This class delivers the average value for coordinate x and y."""

    def __init__(self):
        self.bunchtwissanalysis = BunchTwissAnalysis()
        self.xAvg = 0.0
        self.yAvg = 0.0
        self.xpAvg = 0.0
        self.ypAvg = 0.0

    def analyzeSignal(self, bunch):

        self.bunchtwissanalysis.analyzeBunch(bunch)

        # if mpi operations are enabled, this section of code will
        # determine the rank of the present node
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)

        # only the primary node needs to output the calculated information
        if rank == 0:
            self.xAvg = self.bunchtwissanalysis.getAverage(0)
            self.xpAvg = self.bunchtwissanalysis.getAverage(1)
            self.yAvg = self.bunchtwissanalysis.getAverage(2)
            self.ypAvg = self.bunchtwissanalysis.getAverage(3)

    def getSignalX(self):
        return self.xAvg

    def getSignalXP(self):
        return self.xpAvg

    def getSignalY(self):
        return self.yAvg

    def getSignalYP(self):
        return self.ypAvg
    

def get_bunch_coords(bunch, axis=None, transformer=None, sample=None):
    """Return the phase space coordinate array.

    Parameters
    ----------
    axis : tuple
        The dimensions to keep. For example, `axis=(0, 1, 2, 3)` keeps the
        transverse coordinates. The default is to keep all dimensions.
    transformer : callable
        Transforms the six-dimensional coordinate array.
    sample : int or float
        If greater than 1 and less than the size of the bunch, randomly select
        this many particles.
    """
    if axis is None:
        axis = tuple(range(6))
    X = np.zeros((bunch.getSize(), 6))
    for i in range(X.shape[0]):
        X[i, :] = [
            bunch.x(i),
            bunch.xp(i),
            bunch.y(i),
            bunch.yp(i),
            bunch.z(i),
            bunch.dE(i),
        ]
    if transformer is not None:
        X = transformer(X)
    if sample is not None:
        if 0.0 < sample < 1.0:
            sample = sample * X.shape[0]
        sample = int(min(sample, X.shape[0]))
        idx = np.random.choice(X.shape[0], sample, replace=False)
        X = X[idx, :]
    return X[:, axis]


def get_bunch_centroid(bunch, bunch_twiss_analysis=None):
    if bunch_twiss_analysis is None:
        bunch_twiss_analysis = BunchTwissAnalysis()
    bunch_twiss_analysis.analyzeBunch(bunch)
    _mpi_rank = 0
    _mpi_init = orbit_mpi.MPI_Initialized()
    if orbit_mpi.MPI_Initialized():
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    if _mpi_rank == 0:
        centroid = np.zeros(6)
        for i in range(6):
            centroid[i] = bunch_twiss_analysis.getAverage(i)
        return centroid


def get_bunch_cov(
    bunch, dispersion_flag=False, emit_norm_flag=False, bunch_twiss_analysis=None
):
    if bunch_twiss_analysis is None:
        bunch_twiss_analysis = BunchTwissAnalysis()
    dispersion_flag = int(dispersion_flag)
    emit_norm_flag = int(emit_norm_flag)
    order = 2
    bunch_twiss_analysis.computeBunchMoments(
        bunch, order, dispersion_flag, emit_norm_flag
    )
    _mpi_rank = 0 
    if orbit_mpi.MPI_Initialized():
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    if _mpi_rank == 0:
        cov = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                cov[i, j] = bunch_twiss_analysis.getCorrelation(i, j)
        return cov


class DumpBunchNode(DriftTEAPOT):
    def __init__(self, filename="bunch.dat", name="write_bunch_coords", verbose=False):
        DriftTEAPOT.__init__(self, name)
        self.filename = filename
        self.active = True
        self.verbose = verbose

    def track(self, params_dict):
        if self.active:
            if self.verbose:
                _mpi_rank = 0 
                if orbit_mpi.MPI_Initialized():
                    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
                    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
                if _mpi_rank == 0:
                    print("Writing bunch coordinates to file '{}'".format(self.filename))
            bunch = params_dict["bunch"]
            bunch.dumpBunch(self.filename)


class TBTDiagnosticsNode(DriftTEAPOT):
    """Turn-by-turn diagnostics node.

    Attributes
    ----------
    name : str
        The name of the node.
    ntype : str
        The type of the node.
    position : float
        The position of the node in the lattice [m].
    lattlength : float
        [...]
    active : bool
        Turns the node on/off.
    remember : bool
        Whether to remember previous turns.
    skip : int
        Skip this many turns between each measurement.
    data : list
        Records output of previous measurements.
    turns : list[int]
        The turn number corresponding to each element of `data`.
    turn : int
        The current turn number; i.e., the number of times the bunch has passed this
        node.
    """

    def __init__(
        self,
        name="",
        ntype=None,
        position=None,
        lattlength=0.0,
        active=True,
        remember=True,
        skip=0,
    ):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        if ntype:
            self.setType(ntype)
        self.position = position
        self.lattlength = lattlength  # What is this for?
        self.data, self.turns = [], []
        self.turn = 0
        self.skip = skip
        self.active = active
        self.remember = remember

    def track(self, params_dict):
        """Track the bunch."""
        bunch = params_dict["bunch"]
        if self.should_measure():
            self.data.append(self.measure(bunch))
            self.turns.append(self.turn)
        self.turn += 1

    def should_measure(self):
        """Should be a measurement be performed right now?"""
        if self.measure is None:
            return False
        if not self.active:
            return False
        if self.turn > 0 and self.turn % (self.skip + 1) != 0:
            return False
        return True

    def clear_data(self):
        """Clear stored data (without resetting the turn counter)."""
        if type(self.remember) is int and self.remember > 0:
            self.data = self.data[-self.remember :]
            self.turns = self.turns[-self.remember :]
        elif not self.remember:
            self.data = []
            self.turns = []

    def measure(self, bunch):
        return

    def package_data(self):
        return


class BunchCoordsNode(TBTDiagnosticsNode):
    def __init__(self, axis=None, transformer=None, sample=None, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)
        self.axis = axis
        if self.axis is None:
            self.axis = tuple(range(6))
        self.transformer = transformer
        self.sample = sample

    def measure(self, bunch):
        return get_bunch_coords(
            bunch, axis=self.axis, transformer=self.transformer, sample=self.sample
        )


class DanilovBunchCoordsNode(TBTDiagnosticsNode):
    def __init__(self, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)

    def measure(self, bunch):
        """Measure the envelope parameters and test bunch coordinates.

        Parameters
        ----------
        bunch : Bunch
            The eight envelope parameters are stored in the transverse coordinates
            of the first two particles in the bunch. Any remaining particles are
            test particles. Test particles respond to the field generated by the
            uniform density core, but are causally inert.

        Returns
        -------
        dict
            'params' : ndarray, shape (8,)
                The transverse envelope parameters (a, b, a', b', e, f, e', f'). The phase
                space coordinates of a particle on the envelope are parameterized as:
                    x = a*cos(psi) + b*sin(psi),
                    y = e*cos(psi) + f*sin(psi),
                    x' = a'*cos(psi) + b'*sin(psi),
                    y' = e'*cos(psi) + f'*sin(psi),
                where 0 <= psi <= 2pi.
            'X': ndarray, shape (k - 2, 6)
                The phase space coordinates of the test particles. Each test particle
                responds to the space charge field defined by the envelope. Test
                particles do not affect each other or the envelope.
        """
        X_bunch = get_bunch_coords(bunch)
        a, ap, e, ep = X_bunch[0, :4]
        b, bp, f, fp = X_bunch[1, :4]
        params = np.array([a, b, ap, bp, e, f, ep, fp])
        X = None
        if X_bunch.shape[0] > 2:
            X = X_bunch[2:, :]
        return {"params": params, "X": X}


class FirstOrderMomentsNode(TBTDiagnosticsNode):
    def __init__(self, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)
        self.bunch_twiss_analysis = BunchTwissAnalysis()

    def measure(self, bunch):
        return get_bunch_centroid(bunch, bunch_twiss_analysis=self.bunch_twiss_analysis)

    def package_data(self):
        self.data = np.vstack(self.data)


class SecondOrderMomentsNode(TBTDiagnosticsNode):
    def __init__(self, dispersion_flag=False, emit_norm_flag=False, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)
        self.bunch_twiss_analysis = BunchTwissAnalysis()
        self.dispersion_flag = int(dispersion_flag)
        self.emit_norm_flag = int(emit_norm_flag)
        self.order = 2

    def measure(self, bunch):
        return get_bunch_cov(
            bunch,
            dispersion_flag=self.dispersion_flag,
            emit_norm_flag=self.emit_norm_flag,
            bunch_twiss_analysis=self.bunch_twiss_analysis,
        )

    def package_data(self):
        self.data = np.array(self.data)


class TeapotTuneAnalysisNode(DriftTEAPOT):
    """Measures the transverse tunes."""

    def __init__(self, name="tune_analysis"):
        DriftTEAPOT.__init__(self, name)
        self.bunch_tune_analysis = BunchTuneAnalysis()
        self.setType("tune calculator teapot")
        self.lattlength = 0.0
        self.setLength(0.0)
        self.position = 0.0

    def assign_twiss(
        self,
        beta_x=None,
        alpha_x=None,
        eta_x=None,
        etap_x=None,
        beta_y=None,
        alpha_y=None,
    ):
        self.bunch_tune_analysis.assignTwiss(
            beta_x, alpha_x, eta_x, etap_x, beta_y, alpha_y
        )

    def track(self, params_dict):
        bunch = params_dict["bunch"]
        self.bunch_tune_analysis.analyzeBunch(bunch)


# An following is an old class that I may delete:
class Wire:
    """Represents a single wire.

    Attributes
    ----------
    n_steps : int
        Number of steps in the scan.
    limits : (min, max)
        Minimum and maximum position of the wire.
    rms_frac_count_err : float
        Fractional error in bin counts.
    pos : ndarray, shape (nsteps,)
        Positions of the wire during the scan.
    edges : ndarray, shape (nsteps + 1,)
        Histogram bin edges. We must compute a histogram to simulate the measurement.
    signal : ndarray
        Signal amplitude at each position.
    """

    def __init__(self, n_steps, limits, rms_frac_count_err=None):
        self.n_steps = n_steps
        self.limits = limits
        self.pos = np.linspace(limits[0], limits[1], n_steps)
        self.delta = abs(np.diff(self.pos)[0])
        self.edges = np.hstack(
            [self.pos - 0.5 * self.delta, [self.pos[-1] + 0.5 * self.delta]]
        )
        self.rms_frac_count_err = rms_frac_count_err
        self.signal = []

    def scan(self, data):
        """Record a histogram of the data array. (Overwrites stored data.)"""
        self.signal, _ = np.histogram(data, bins=self.edges)
        if self.rms_frac_count_err:
            noise = np.random.normal(scale=self.rms_frac_count_err, size=self.n_steps)
            self.signal = self.signal * (1.0 + noise)
            self.signal = self.signal.astype(int)
            self.signal = np.clip(self.signal, 0.0, None)

    def mean(self):
        """Estimate the mean of the data from the histogram."""
        N = np.sum(self.signal)
        return np.sum(self.signal * self.pos) / (N - 1)

    def var(self):
        """Estimate the variance of the data from the histogram."""
        N = np.sum(self.signal)
        x_avg = self.mean()
        x2_avg = np.sum(self.signal * self.pos**2) / (N - 1)
        return x2_avg - x_avg**2

    def std(self):
        """Estimate the standard deviation of the data from the histogram."""
        return np.sqrt(self.var())


class WireScannerNode(DriftTEAPOT):
    """Represents a wire-scanner.

    Attributes
    ----------
    n_steps : int
        Number of steps in the scan. Default is 90 (true for SNS RTBT wire-
        scanners).
    ulims : (umin, umax)
        Minimum and maximum position of the wire along the u axis [m]. Default
        is (-0.1335, 0.1335) for SNS RTBT wire-scanners).
    phi : float
        Angle of the u axis -- the axis perpendicular to the diagonal wire --
        above the x axis [rad]. Default is pi/4 (for SNS RTBT wire-scanners).
    rms_frac_count_err : float
        RMS fractional error in bin counts. The counts are kept above zero.
    x, y, u : Wire
        The three wires in the device: vertical (x), horizontal (y), and
        diagonal (u).
    """

    def __init__(
        self,
        n_steps=None,
        ulims=None,
        phi=None,
        rms_frac_count_err=None,
        name="wire-scanner",
    ):
        DriftTEAPOT.__init__(self, name)
        if n_steps is None:
            n_steps = 90
        if ulims is None:
            ulims = (-0.1335, 0.1335)
        if phi is None:
            phi = np.radians(45.0)
        self.n_steps = n_steps
        self.rms_frac_count_err = rms_frac_count_err
        self.phi = phi
        umin, umax = ulims
        xmin = umin * np.cos(self.phi)
        xmax = umax * np.cos(self.phi)
        ymin = umin * np.sin(self.phi)
        ymax = umax * np.sin(self.phi)
        self.xlims = (xmin, xmax)
        self.ylims = (ymin, ymax)
        self.ulims = (umin, umax)
        self.xwire = Wire(self.n_steps, self.xlims, self.rms_frac_count_err)
        self.ywire = Wire(self.n_steps, self.ylims, self.rms_frac_count_err)
        self.uwire = Wire(self.n_steps, self.ulims, self.rms_frac_count_err)
        self.wires = [self.xwire, self.ywire, self.uwire]

    def track(self, params_dict):
        """Track and compute histograms. (Overwrites stored data.)"""
        bunch = params_dict["bunch"]
        X = bunch_coord_array(bunch)
        xx = X[:, 0]
        yy = X[:, 2]
        uu = xx * np.cos(self.phi) + yy * np.sin(self.phi)
        self.xwire.scan(xx)
        self.ywire.scan(yy)
        self.uwire.scan(uu)

    def get_moments(self, assumed_angle=None):
        """Estimate cov(x, x), cov(y, y) and cov(x, y).

        If `assumed_angle` is provided, use it for the angle of the u axis
        instead of the true angle.
        """
        sig_xx = self.xwire.var()
        sig_yy = self.ywire.var()
        sig_uu = self.uwire.var()
        phi = self.phi
        if assumed_angle is not None:
            phi = assumed_angle
        cs, sn = np.sin(phi), np.cos(phi)
        sig_xy = (sig_uu - sig_xx * cs**2 - sig_yy * sn**2) / (2 * sn * cs)
        return sig_xx, sig_yy, sig_xy
