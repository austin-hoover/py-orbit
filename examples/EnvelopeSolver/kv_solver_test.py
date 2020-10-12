"""Benchmark of KV envelope solver.

This script tracks the 2D KV distribution through a FODO cell using the 2.5D 
space charge model and compares the results with the envelope model.
"""

# Imports
#------------------------------------------------------------------------------
import numpy as np

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D, EnvSolverKV
from orbit.bunch_generators import TwissContainer, KVDist2D
from orbit.teapot import teapot, TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.space_charge.envelope import setEnvAccNodes

from utils import (
    create_lattice, lattice_twiss, split_nodes, 
    initialize_bunch, get_perveance, radii_from_twiss, cov_mat
)

# Settings
#------------------------------------------------------------------------------

# Beam
nparts = int(1e5)
nturns = 10
mass = 0.93827231 # [GeV/c^2]
energy = 1.0 # [GeV]
eps_x = eps_y = 25e-6 # rms emittance (m*rad)
intensity = 2.0e14

# Lattice
lattice_file = 'fodo.lat'
seq = 'fodo'

# Space charge solver
max_solver_spacing = 0.1
sc_path_length_min = 0.0000001
gridpts_x = 128 # number of grid points in horizontal direction
gridpts_y = 128 # number of grid points in vertical direction
gridpts_z = 1 # number of longitudinal slices


# Envelope solver
#------------------------------------------------------------------------------

# Create FODO lattice
lattice = create_lattice(lattice_file, seq)
lattice_length = lattice.getLength()
alpha_x, beta_x, alpha_y, beta_y = lattice_twiss(lattice, mass, energy)

# Add space charge solver nodes
split_nodes(lattice, max_solver_spacing)
Q = get_perveance(energy, mass, intensity/lattice_length)
setEnvAccNodes(lattice, sc_path_length_min, EnvSolverKV(eps_x, eps_y, Q))

# Create bunch
a, ap = radii_from_twiss(alpha_x, beta_x, eps_x)
b, bp = radii_from_twiss(alpha_y, beta_y, eps_y)
env, params_dict_env = initialize_bunch(mass, energy)
env.addParticle(a, ap, b, bp, 0.0, 0.0)

# Track envelope
env_dims = np.zeros((nturns + 1, 2))
for i in range(nturns + 1):
    env_dims[i] = 1e6 * np.array([env.x(0)**2, env.y(0)**2]) / 4
    lattice.trackBunch(env, params_dict_env)    
    
    
# FFT solver
#------------------------------------------------------------------------------

# Create FODO lattice
lattice = create_lattice(lattice_file, seq)

# Add space charge solver onodes
split_nodes(lattice, max_solver_spacing)
setSC2p5DAccNodes(lattice, sc_path_length_min, 
                  SpaceChargeCalc2p5D(gridpts_x, gridpts_y, gridpts_z))

# Create bunch
dist = KVDist2D(
    TwissContainer(alpha_x, beta_x, eps_x),
    TwissContainer(alpha_y, beta_y, eps_y)
)
bunch, params_dict = initialize_bunch(mass, energy)
bunch.macroSize(intensity / nparts)
for _ in range(nparts): 
    x, xp, y, yp = dist.getCoordinates()
    z = np.random.random() * lattice_length
    bunch.addParticle(x, xp, y, yp, z, 0.0)
    
# Track
print ''
print 'turn | <x^2>_beam | <x^2>_env  | <y^2>_beam | <y^2>_env' 
print '-------------------------------------------------------'
f = '{:<4} | {:<10.3f} | {:<10.3f} | {:<10.3f} | {:<10.3f}'

beam_dims = np.zeros((nturns + 1, 2))

for i in range(nturns + 1):
    sigma = cov_mat(bunch)
    beam_dims[i] = 1e6 * np.array([sigma[0, 0], sigma[2, 2]])
    lattice.trackBunch(bunch, params_dict)
    print f.format(
        i, 
        beam_dims[i, 0], env_dims[i, 0], 
        beam_dims[i, 1], env_dims[i, 1], 
    )
    
# Uncomment below to plot results
#------------------------------------------------------------------------------

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
for i, ax in enumerate(axes):
    ax.plot(env_dims[:, i], 'k--', lw=0.5)
    ax.plot(beam_dims[:, i], marker='+', lw=0, color='red')
    ax.set_xlabel('Turn number')
axes[0].set_ylabel(r'[${mm}^2$]')
axes[0].set_title(r'$\langle{x^2}\rangle$')
axes[1].set_title(r'$\langle{y^2}\rangle$')
axes[1].legend(labels=['Envelope', 'FFT'])
fig.set_tight_layout(True)
plt.savefig('kv_benchmark.png', dpi=200)

print "Results shown in 'kv_benchmark.png'"