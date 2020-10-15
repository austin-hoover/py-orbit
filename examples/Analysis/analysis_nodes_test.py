"""Test of the analysis nodes."""

import os
import random

from bunch import Bunch
from orbit.analysis import add_analysis_nodes
from orbit.bunch_generators import TwissContainer, KVDist2D
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

os.system('rm ./_output/*')

#--------------------------------------------------------------------

# Settings
output_dir = '_output'
max_analysis_node_sep = 0.01
nparts = int(1e3)
mass = 0.93827231 # [GeV/c^2]
energy = 1.0 # [GeV]
ex = 15e-6 # rms emittance [m*rad]
ey = 35e-6

# Create lattice
lattice = hf.lattice_from_file('fodo.lat', 'fodo')
hf.split_nodes(lattice, 0.01)
ax, ay, bx, by = hf.twiss_at_injection(lattice, mass, energy)
analysis_nodes  = add_analysis_nodes(
    lattice, output_dir, max_analysis_node_sep
)

# Create distribution
dist = KVDist2D(TwissContainer(ax, bx, ex), TwissContainer(ay, by, ey))
bunch, params_dict = hf.initialize_bunch(mass, energy)
for _ in range(nparts):
    x, xp, y, yp = dist.getCoordinates()
    z = random.random() * lattice.getLength()
    bunch.addParticle(x, xp, y, yp, z, 0.0)

lattice.trackBunch(bunch, params_dict)