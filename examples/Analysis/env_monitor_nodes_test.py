"""Test of the envelope monitor nodes."""

import os
import random

from bunch import Bunch
from orbit.analysis import EnvMonitorNode, add_monitor_nodes
from orbit.bunch_generators import DanilovDist2D
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

os.system('rm ./_output/*')

#--------------------------------------------------------------------

# Settings
filename = '_output/env_params.dat'
max_monitor_node_sep = 0.01
nparts = int(1e3)
mass = 0.93827231 # [GeV/c^2]
energy = 1.0 # [GeV]
ex = 15e-6 # rms emittance [m*rad]
ey = 35e-6

# Create lattice
lattice = hf.lattice_from_file('fodo.lat', 'fodo')
hf.split_nodes(lattice, 0.01)
ax, ay, bx, by = hf.twiss_at_injection(lattice, mass, energy)
monitor_nodes  = add_monitor_nodes(
    lattice, filename, EnvMonitorNode, max_monitor_node_sep
)

# Create envelope
dist = DanilovDist2D((ax, bx, ex), (ay, by, ey))
bunch, params_dict = hf.initialize_envelope(dist.params, mass, energy)

lattice.trackBunch(bunch, params_dict)