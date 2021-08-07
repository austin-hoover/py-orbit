import numpy as np

from bunch import Bunch
from analysis import bunch_coord_array
from analysis import BunchStats
from analysis import DanilovEnvelopeBunch
from orbit.teapot import DriftTEAPOT


class AnalysisNode(DriftTEAPOT):
    
    def __init__(self, name):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        self.position = None
        self.kind = None
        self.data = []
        
    def set_position(self, position):
        self.position = position
        
    def track(self, params_dict):
        return
        
    def get_data(self, turn='all'):
        if turn == 'all':
            return self.data
        return self.data[turn]
    
    def clear_data(self):
        self.data = []
    
                    
class BunchMonitorNode(AnalysisNode):
    
    def __init__(self, name='bunch_monitor', mm_mrad=False, transverse_only=True):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        self.transverse_only = transverse_only
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, self.transverse_only)
        self.data.append(X)
        
    
class BunchStatsNode(AnalysisNode):
    
    def __init__(self, name='bunch_stats', mm_mrad=False):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
        self.data.append(BunchStats(X))
        
        
class DanilovEnvelopeBunchMonitorNode(AnalysisNode):
    
    def __init__(self, name='envelope_monitor', mm_mrad=False):
        AnalysisNode.__init__(self, name)
        self.mm_mrad = mm_mrad
        
    def track(self, params_dict):
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, self.mm_mrad, transverse_only=True)
        self.data.append(DanilovEnvelopeBunch(X))
        
        
        
class WireScanner1D(AnalysisNode):

    def __init__(self, n_steps, range, frac_count_err=None):
        self.n_steps = n_steps
        self.range = range
        self.frac_count_err = frac_count_err
        self.pos = []
        self.raw = []

    def bin_data(self, data):
        self.raw, bin_edges = np.histogram(data, bins=self.n_steps, range=self.range)
        delta = np.mean(np.diff(bin_edges))
        self.pos = (bin_edges + 0.5 * delta)[:-1]
        if self.frac_count_err:
            lo = self.raw * (1 - self.frac_count_err)
            hi = self.raw * (1 + self.frac_count_err)
            noise = np.random.uniform(lo, hi).astype(int)
            self.raw = np.clip(self.raw + noise, 0, None)

    def mean(self):
        N = np.sum(self.raw)
        return np.sum(self.raw * self.pos) / (N - 1)

    def var(self):
        N = np.sum(self.raw)
        x_avg = self.mean()
        x2_avg = np.sum(self.raw * self.pos**2) / (N - 1)
        return x2_avg - x_avg**2 


class WireScannerNode(AnalysisNode):
    """Node to measure simulate wire-scanner measurement.
    
    Attributes
    ----------
    n_steps : int
        The number of steps in the scan. Default is 90, which is true in the
        SNS RTBT wire-scanners.
    urange : (u_min, u_max)
        The initial and final position along the u axis [mm]. Then x and y
        are given by x = u * cos(phi) and y = u * sin(phi), where phi is
        the angle of the u axis above the x axis (phi = diag_wire_angle + pi/2).
        Default is (25.0, 292.0) [mm], which is true in the SNS RTBT.
    diag_wire_angle : float
        Diagonal wire angle [rad]. Default is -45.0, which is true in the SNS 
        RTBT wire-scanners.
    phi : float
        Angle of u axis above x axis (diag_wire_angle + 90 degrees).
    tilt_err : float
        Error in angle of diagonal wire [rad]. We use diag_wire_angle in 
        our calculations, but the angle along which to actually project the 
        distribution when calculating the histogram is randomly chosen in the 
        range [diag_wire_angle - tilt_err, diag_wire_angle + tilt_err].
    frac_count_err : float
        Fractional error in bin counts. Each bin count C is updated to a random
        number in the range [C * (1 - frac_err), C * (1 + frac_err)]. For now,
        the counts are always kept above zero.
    """
    def __init__(self, n_steps=None, urange=None, diag_wire_angle=None,
                 tilt_err=None, frac_count_err=None, name='wire-scanner'):
        DriftTEAPOT.__init__(self, name)  
        if n_steps is None:
            n_steps = 90
        if urange is None:
            urange = (25.0, 292.0)
        if diag_wire_angle is None:
            diag_wire_angle = -0.25 * np.pi
        self.n_steps = n_steps
        self.diag_wire_angle = diag_wire_angle
        self.frac_count_err = frac_count_err
        self.tilt_err = tilt_err
        self.phi = diag_wire_angle + np.radians(90)
        umin, umax = urange
        xmin = umin * np.cos(self.phi)
        xmax = umax * np.cos(self.phi)
        ymin = umin * np.sin(self.phi)
        ymax = umax * np.sin(self.phi)
        self.xrange = (xmin, xmax)
        self.yrange = (ymin, ymax)
        self.urange = (umin, umax)
        self.hor = WireScanner1D(self.n_steps, self.xrange, self.frac_count_err)
        self.ver = WireScanner1D(self.n_steps, self.yrange, self.frac_count_err)
        self.dia = WireScanner1D(self.n_steps, self.urange, self.frac_count_err)
        self.wires = [self.hor, self.ver, self.dia]
        
    def track(self, params_dict):
        """Track and compute histograms. Overwrites previous scan."""
        
        bunch = params_dict['bunch']
        X = bunch_coord_array(bunch, transverse_only=True, mm_mrad=True)
        xx = X[:, 0]
        yy = X[:, 1]
        uu = xx * np.cos(self.phi) + yy * np.sin(self.phi)        
        self.hor.bin_data(xx)
        self.ver.bin_data(yy)
        self.dia.bin_data(uu)
        
    def get_moments(self):
        """Get xx, yy, and xy covariances from histograms."""
        sig_xx = self.hor.var()
        sig_yy = self.ver.var()
        sig_uu = self.dia.var()
        sn = np.sin(self.phi)
        cs = np.cos(self.phi)
        sig_xy = (sig_uu - sig_xx*cs**2 - sig_yy*sn**2) / (2 * sn * cs)
        return sig_xx, sig_yy, sig_xy