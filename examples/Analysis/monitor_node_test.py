"""Test of the one particle monitor nodes."""

import os
import math

from bunch import Bunch
from orbit.analysis import add_onepart_monitor_nodes
from orbit.bunch_generators import TwissContainer, KVDist2D
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

os.system('rm ./_output/*')

#--------------------------------------------------------------------

# Settings
filename = '_output/one_part_coords.dat'
max_analysis_node_sep = 0.01
mass = 0.93827231 # [GeV/c^2]
energy = 1.0 # [GeV]
ex = 15e-6 # rms emittance [m*rad]
ey = 35e-6

# Create lattice
lattice = hf.lattice_from_file('fodo.lat', 'fodo')
hf.split_nodes(lattice, 0.01)
ax, ay, bx, by = hf.twiss_at_injection(lattice, mass, energy)
analysis_nodes  = add_onepart_monitor_nodes(
    lattice, filename, max_analysis_node_sep
)

# Create one particle bunch
x = math.sqrt(ex * bx)
y = math.sqrt(ey * by)
xp = -ax * math.sqrt(ex / bx)
yp = -ay * math.sqrt(ey / by)
bunch, params_dict = hf.initialize_bunch(mass, energy)
bunch.addParticle(x, xp, y, yp, 0.0, 0.0)

# Track
lattice.trackBunch(bunch, params_dict)