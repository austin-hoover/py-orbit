###############################################################################

# Auxiliary classes
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject

# General accelerator elements and lattice
from orbit.lattice import AccNode, AccActionsContainer, AccNodeBunchTracker

# Teapot drift class
from orbit.teapot import DriftTEAPOT
 
class EnvParamsWriter:
    
    def __init__(self, filename):
        self.file = open(filename, 'a')
    
    def write(self, bunch, position, period=0, latt_len=0.0):
        position += period * latt_len
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        f = 8 * '{} ' + '{}\n'
        self.file.write(f.format(position, a, b, ap, bp, e, f, ep, fp))
        
class OnePartWriter:
    
    def __init__(self, filename):
        self.file = open(filename, 'a')
    
    def write(self, bunch, position, period=0, latt_len=0.0):
        position += period * latt_len
        x, xp, y, yp = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        f = 4 * '{} ' + '{}\n'
        self.file.write(f.format(position, x, xp, y, yp))
    
        
class EnvMonitorNode(DriftTEAPOT):

    def __init__(self, file, position, name='env_monitor_no_name'):
        DriftTEAPOT.__init__(self,name)
        self.writer = EnvParamsWriter(file)
        self.position = position
        self.setLength(0.0)

    def track(self, params_dict):
        bunch = params_dict['bunch']
        self.writer.write(bunch, self.position)
        
    def set_position(self, position):
        self.position = position

    def close(self):
        self.file.close()


class OnePartMonitorNode(DriftTEAPOT):

    def __init__(self, file, position, name='one_part_monitor_no_name'):
        DriftTEAPOT.__init__(self, name)
        self.writer = OnePartWriter(file)
        self.position = position
        self.setLength(0.0)

    def track(self, params_dict):
        bunch = params_dict['bunch']
        self.writer.write(bunch, self.position)
        
    def set_position(self, position):
        self.position = position

    def close(self):
        self.file.close()
