
import numpy as np

from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker
from orbit.teapot import DriftTEAPOT
 
 
class EnvWriter:

    def __init__(self, filename_env, filename_testbunch):
        self.filename_env = open(filename_env, 'a')
        self.filename_testbunch = filename_testbunch

    def write(self, bunch, position):

        # Envelope parameters (first two bunch coordinates)
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        form = 8 * '{} ' + '{}\n'
        self.filename_env.write(form.format(
            position, a, b, ap, bp, e, f, ep, fp))

        # Transverse coordinates of test bunch
        n = bunch.getSize()
        if n > 2:
            X = []
            for i in range(2, n):
                X.append([bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)])
            np.savetxt(self.filename_testbunch, np.array(X))


        
class OnePartWriter:
    
    def __init__(self, filename):
        self.file = open(filename, 'a')
    
    def write(self, bunch, position):
        x, xp, y, yp = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        form = 4 * '{} ' + '{}\n'
        self.file.write(form.format(position, x, xp, y, yp))
    
        
class EnvMonitorNode(DriftTEAPOT):

    def __init__(self, path_env, path_testbunch, position, name='env_monitor', tbt=False):
        DriftTEAPOT.__init__(self, name)
        self.turn_idx = 0
        if not path_env.endswith('/'):
            path_env += '/'
        if not path_testbunch.endswith('/'):
            path_testbunch += '/'
        self.path_testbunch = path_testbunch
        self.path_env = path_env
        filename_env = ''.join([path_env, 'env_params.dat'])
        if tbt:
            filename_testbunch = ''.join([path_testbunch, 'coords_{}.dat'.format(self.turn_idx)])
        else:
            filename_testbunch = ''.join([path_testbunch, 'coords_s{:.4f}.dat'.format(position)])
        self.writer = EnvWriter(filename_env, filename_testbunch)
        self.position = position
        self.setLength(0.0)

    def track(self, params_dict):
        self.writer.filename_testbunch = ''.join([
            self.path_testbunch, 'coords_{}.dat'.format(self.turn_idx)
        ])
        self.writer.write(params_dict['bunch'], self.position)
        self.turn_idx += 1
        
    def set_position(self, position):
        self.position = position

    def close(self):
        self.file.close()


class OnePartMonitorNode(DriftTEAPOT):

    def __init__(self, file, position, name='one_part_monitor'):
        DriftTEAPOT.__init__(self, name)
        self.writer = OnePartWriter(file)
        self.position = position
        self.setLength(0.0)

    def track(self, params_dict):
        self.writer.write(params_dict['bunch'], self.position, self.tilt)
        
    def set_position(self, position):
        self.position = position

    def close(self):
        self.file.close()
