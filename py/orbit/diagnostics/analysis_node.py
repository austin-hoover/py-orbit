from bunch import Bunch
from analysis import bunch_coord_array
from analysis import BunchStats
from analysis import DanilovEnvelopeBunch
from orbit.teapot import DriftTEAPOT
from orbit.utils.general import apply
from orbit.utils.general import rotation_matrix_4D


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
        

class WireScannerNode(AnalysisNode):
    """Node to measure simulate wire-scanner measurement.
    
    Attributes
    ----------
    nbins : int
        Number of bins in each histogram, i.e., number of steps the wire takes
        on its path across the beam.
    phi : float
        Diagonal wire angle [rad].
    dphi : float
        Error in angle of diagonal wire [rad]. We use phi in our calculations,
        but the angle along which to actually project the distribution when
        calculating the histogram will be randomly chosen in the range
        [phi - dphi, phi + dpi].
    dcount : float
        Fractional error in bin counts. Each bin count C is updated to a random
        number in the range [C * (1 - dcounts), C * (1 + dcounts)].
    """
    def __init__(self, name='wire-scanner'):
        DriftTEAPOT.__init__(self, name)
#     def __init__(self, nbins=50, phi=1.57, name='ws', dphi=0.0, dcount=0.0):
#         DriftTEAPOT.__init__(self, name)
#         self.nbins = 50
#         self.phi = phi
#         self.dphi = dphi
#         self.dcount = dcount
#         self.hist, self.pos = {}, {}
        
#     def set_frac_bin_count_error(self, dcount):
#         self.dcount = dcount
        
#     def set_diag_wire_angle_error(self, dphi):
#         self.dphi = dphi
                
#     def track(self, params_dict):
#         """Track and compute histograms. Overwrites previous scan."""
#         def bin_data(data):
#             counts, bin_edges = np.histogram(data, self.nbins)
#             # Get bin centers
#             delta = np.mean(np.diff(bin_edges))
#             bin_centers = (bin_edges + 0.5 * delta)[:-1]
#             # Add random noise to counts
#             if self.dcount > 0:
#                 lo = counts * (1 - self.dcount)
#                 hi = counts * (1 + self.dcount)
#                 dcounts = np.random.uniform(lo, hi).astype(int)
#                 counts = np.clip(counts + dcounts, 0, None)
#             return counts, bin_centers
            
#         # Vertical and horizontal wire
#         X = coord_array(params_dict['bunch'])
#         self.hist['x'], self.pos['x'] = bin_data(X[:, 0])
#         self.hist['y'], self.pos['y'] = bin_data(X[:, 2])
#         # Diagonal wire: rotate coordinates so that the diagonal wire points
#         # along the y axis. We add a random number to the rotation angle to
#         # simulate the uncertainty in the true angle.
#         angle = self.phi * np.random.uniform((1 - self.dphi), (1 + self.dphi))
#         X = apply(rotation_matrix_4D(angle), X)
#         self.hist['u'], self.pos['u'] = bin_data(X[:, 0])

#     def estimate_variance(self, dim='x'):
#         """Estimate variance from histogram."""
#         counts, positions = self.hist[dim], self.pos[dim]
#         N = np.sum(counts)
#         x_avg = np.sum(counts * positions) / (N - 1)
#         x2_avg = np.sum(counts * positions**2) / (N - 1)
#         return x2_avg - x_avg**2
        
#     def get_moments(self):
#         """Get xx, yy, and xy covariances from histograms."""
#         sig_xx = self.estimate_variance('x')
#         sig_yy = self.estimate_variance('y')
#         sig_uu = self.estimate_variance('u')
#         sin, cos = np.sin(self.phi), np.cos(self.phi)
#         sig_xy = (sig_uu - sig_xx*cos**2 - sig_yy*sin**2) / (2 * sin * cos)
#         return sig_xx, sig_yy, sig_xy
