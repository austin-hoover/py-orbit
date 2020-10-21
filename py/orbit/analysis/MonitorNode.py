
import numpy as np

from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT


def rot_mat(phi):
    cos, sin = np.cos(phi), np.sin(phi)
    R = np.array([
        [cos, 0, sin, 0],
        [0, cos, 0, sin],
        [-sin, 0, cos, 0],
        [0, -sin, 0, cos]
    ])
    return R
 
 
class EnvWriter:
    
    def __init__(self, filename):
        self.file = open(filename, 'a')
    
    def write(self, bunch, position, tilt=0):
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        if tilt > 0:
            P = np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
            R = rot_mat(tilt)
            a, b, ap, bp, e, f, ep, fp = np.matmul(R, P).flatten()
        form = 8 * '{} ' + '{}\n'
        self.file.write(form.format(position, a, b, ap, bp, e, f, ep, fp))

        
class OnePartWriter:
    
    def __init__(self, filename):
        self.file = open(filename, 'a')
    
    def write(self, bunch, position, tilt=0):
        x, xp, y, yp = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        if tilt > 0:
            R = rot_mat(tilt)
            x, xp, y, yp = np.matmul(R, np.array([x, xp, y, yp]))
        form = 4 * '{} ' + '{}\n'
        self.file.write(form.format(position, x, xp, y, yp))
    
        
class EnvMonitorNode(DriftTEAPOT):

    def __init__(self, file, position, name='env_monitor_no_name', tilt=0.0):
        DriftTEAPOT.__init__(self, name)
        self.writer = EnvWriter(file)
        self.position = position
        self.tilt = tilt
        self.setLength(0.0)

    def track(self, params_dict):
        self.writer.write(params_dict['bunch'], self.position, self.tilt)
        
    def set_position(self, position):
        self.position = position
        
    def set_tilt(self, tilt):
        self.tilt = tilt

    def close(self):
        self.file.close()


class OnePartMonitorNode(DriftTEAPOT):

    def __init__(self, file, position, name='one_part_monitor_no_name', tilt=0.0):
        DriftTEAPOT.__init__(self, name)
        self.writer = OnePartWriter(file)
        self.position = position
        self.tilt = tilt
        self.setLength(0.0)

    def track(self, params_dict):
        self.writer.write(params_dict['bunch'], self.position, self.tilt)
        
    def set_position(self, position):
        self.position = position
        
    def set_tilt(self, tilt):
        self.tilt = tilt

    def close(self):
        self.file.close()
