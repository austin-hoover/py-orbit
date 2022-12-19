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
        
    def measure(self, bunch):
        return

    def track(self, params_dict):
        """Track the bunch."""
        bunch = params_dict["bunch"]
        if self.should_measure():
            self.register(self.measure(bunch))
        self.turn += 1

    def should_measure(self):
        """Should be a measurement be performed right now?"""
        if self.measure is None:
            return False
        if not self.active:
            return False
        if self.turn > 0 and self.turn  % (self.skip + 1) != 0:
            return False
        return True
              
    def register(self, item):
        """Store measured item and turn number."""
        self.clear_data()
        self.data.append(item)
        self.turns.append(self.turn)

    def clear_data(self):
        """Clear stored data (without resetting the turn counter)."""
        if type(self.remember) is int and self.remember > 0:
            self.data = self.data[-self.remember:]
            self.turns = self.turns[-self.remember:]
        elif not self.remember:
            self.data = []
            self.turns = []            
        
    def package_data(self):
        return
        

class BunchCoordsNode(TBTDiagnosticsNode):
    """Measure the phase space coordinate array.

    axis : tuple
        The dimensions to keep. For example, `axis=(0, 1, 2, 3)` keeps the transverse coordinates.
        Default is to keep all dimensions.
    transformer : callable
        Transforms the six-dimensional coordinate array.
    sample : int or float
        If greater than 1 and less than the size of the bunch, randomly select
        this many particles.
    """
    def __init__(self, axis=None, transformer=None, sample=None, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)
        self.axis = axis
        if self.axis is None:
            self.axis = tuple(range(6))
        self.transformer = transformer
        self.sample = sample
        
    def measure(self, bunch):
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
        if self.transformer is not None:
            X = self.transformer(X)
        if self.sample is not None:
            sample = self.sample
            if 0.0 < sample < 1.0:
                sample = sample * X.shape[0]
            sample = int(min(sample, X.shape[0]))
            idx = np.random.choice(X.shape[0], sample, replace=False)
            X = X[idx, :]
        return X[:, self.axis]
    

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
        X_bunch = np.zeros((bunch.getSize(), 6))
        for i in range(bunch.getSize()):
            X_bunch[i, :] = [
                bunch.x(i),
                bunch.xp(i),
                bunch.y(i),
                bunch.yp(i),
                bunch.z(i),
                bunch.dE(i),
            ]
        a, ap, e, ep = X_bunch[0, :4]
        b, bp, f, fp = X_bunch[1, :4]
        params = np.array([a, b, ap, bp, e, f, ep, fp])
        X = None
        if X_bunch.shape[0] > 2:
             X = X_bunch[2:, :]
        return {'params': params, 'X': X}
        
        
class FirstOrderMomentsNode(TBTDiagnosticsNode):
    def __init__(self, **kws):
        TBTDiagnosticsNode.__init__(self, **kws)
        self.bunch_twiss_analysis = BunchTwissAnalysis()
        
    def measure(self, bunch):
        self.bunch_twiss_analysis.analyzeBunch(bunch)
        # Determine the rank of the present node if MPI operations are enabled.
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)
        # Only the primary node needs to output the calculated information.
        if rank == 0:
            centroid = np.zeros(6)
            for i in range(6):
                centroid[i] = self.bunch_twiss_analysis.getAverage(i)
            return centroid
        
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
        self.bunch_twiss_analysis.computeBunchMoments(bunch, self.order, self.dispersion_flag, self.emit_norm_flag)
        # Determine the rank of the present node if MPI operations are enabled.
        rank = 0  # default is primary node
        mpi_init = orbit_mpi.MPI_Initialized()
        comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        if mpi_init:
            rank = orbit_mpi.MPI_Comm_rank(comm)
        # Only the primary node needs to output the calculated information.
        if rank == 0:
            cov = np.zeros((6, 6))
            for i in range(6):
                for j in range(i + 1):
                    cov[i, j] = self.bunch_twiss_analysis.getCorrelation(i, j)
            return cov
        
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