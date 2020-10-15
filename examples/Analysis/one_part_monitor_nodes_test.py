"""Test of the one particle monitor nodes."""

import os
from math import sqrt

from bunch import Bunch
from orbit.analysis import OnePartMonitorNode, add_monitor_nodes
from orbit.bunch_generators import TwissContainer, KVDist2D
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

os.system('rm ./_output/*')

#--------------------------------------------------------------------

# Settings
filename = '_output/one_part_coords.dat'
max_monitor_node_sep = 0.01
mass = 0.93827231 # [GeV/c^2]
energy = 1.0 # [GeV]
ex = 15e-6 # rms emittance [m*rad]
ey = 35e-6

# Create lattice
lattice = hf.lattice_from_file('fodo.lat', 'fodo')
hf.split_nodes(lattice, 0.01)
ax, ay, bx, by = hf.twiss_at_injection(lattice, mass, energy)
monitor_nodes  = add_monitor_nodes(
    lattice, filename, OnePartMonitorNode, max_monitor_node_sep
)

# Create one particle bunch
x = sqrt(ex * bx)
y = sqrt(ey * by)
xp = -ax * sqrt(ex / bx)
yp = -ay * sqrt(ey / by)
bunch, params_dict = hf.initialize_bunch(mass, energy)
bunch.addParticle(x, xp, y, yp, 0.0, 0.0)

# Track
lattice.trackBunch(bunch, params_dict)
