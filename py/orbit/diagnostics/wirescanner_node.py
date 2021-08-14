import numpy as np

from bunch import Bunch
from analysis import bunch_coord_array
from orbit.teapot import DriftTEAPOT


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
    pos : ndarray
        Positions of the wire during the scan.
    signal : ndarray
        Signal amplitude at each position.
    """
    def __init__(self, n_steps, limits, rms_frac_count_err=None):
        self.n_steps = n_steps
        self.limits = limits
        self.rms_frac_count_err = rms_frac_count_err
        self.pos = []
        self.signal = []

    def scan(self, data):
        """Record a histogram of the data array. (Overwrites stored data.)"""
        self.signal, bin_edges = np.histogram(data, self.n_steps, self.limits)
        delta = np.mean(np.diff(bin_edges))
        self.pos = (bin_edges + 0.5 * delta)[:-1]
        if self.rms_frac_count_err:
            noise = np.random.normal(scale=self.rms_frac_count_err, size=len(self.signal))
            self.signal = self.signal * (1.0 + noise)
            self.signal = self.signal.astype(int)
            self.signal = np.clip(self.signal, 0, None)

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
    """Represents a wire-scanner device.

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
    def __init__(self, n_steps=None, ulims=None, phi=None, rms_frac_count_err=None,
                 name='wire-scanner'):
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
        bunch = params_dict['bunch']
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
        if assumed_angle:
            phi = assumed_angle
        cs, sn = np.sin(phi), np.cos(phi)
        sig_xy = (sig_uu - sig_xx*cs**2 - sig_yy*sn**2) / (2 * sn * cs)
        return sig_xx, sig_yy, sig_xy