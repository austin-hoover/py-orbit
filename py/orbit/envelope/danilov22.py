"""Envelope model for the {2, 2} Danilov distribution.

References
----------
[1] https://doi.org/10.1103/PhysRevSTAB.6.094202
[2] https://doi.org/10.1103/PhysRevAccelBeams.24.044201
"""
from __future__ import print_function
import copy
import time

import numpy as np
import scipy.optimize as opt
from tqdm import trange

from bunch import Bunch
from orbit.matrix_lattice.parameterizations import courant_snyder as CS
from orbit.matrix_lattice.parameterizations import lebedev_bogacz as LB
from orbit.matrix_lattice.transfer_matrix import TransferMatrix
from orbit.matrix_lattice.transfer_matrix import TransferMatrixCourantSnyder
from orbit.matrix_lattice.transfer_matrix import TransferMatrixLebedevBogacz
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils import consts


def rotation_matrix(angle):
    """2 x 2 clockwise rotation matrix (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4x4(angle):
    """4 x 4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_advance_matrix(*phase_advances):
    """Phase advance matrix (clockwise rotation in each phase plane).

    Parameters
    ---------
    mu1, mu2, ..., mun : float
        The phase advance in each plane.

    Returns
    -------
    ndarray, shape (2n, 2n)
        Matrix which rotates x-x', y-y', z-z', etc. by the phase advances.
    """
    n = len(phase_advances)
    M = np.zeros((2 * n, 2 * n))
    for i, phase_advance in enumerate(phase_advances):
        i = i * 2
        M[i : i + 2, i : i + 2] = rotation_matrix(phase_advance)
    return M


def initialize_bunch(mass=None, kin_energy=None):
    """Create and initialize a Bunch.

    Parameters
    ----------
    mass, energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per bunch particle.

    Returns
    -------
    bunch : Bunch
        A Bunch object with the given mass and kinetic energy.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    params_dict = {"bunch": bunch}
    return bunch, params_dict


def get_transfer_matrix(lattice, mass, energy):
    """Return linear 6x6 transfer matrix from periodic lattice as ndarray.

    Parameters
    ----------
    lattice : TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].

    Returns
    -------
    M : ndarray, shape (6, 6)
        Transverse transfer matrix.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            M[i, j] = matrix_lattice.oneTurnMatrix.get(i, j)
    return M


def get_moment_vector(Sigma):
    """Return array of 10 unique elements of covariance matrix."""
    return Sigma[np.triu_indices(4)]


def get_perveance(mass, kin_energy, line_density):
    """ "Compute dimensionless beam perveance.

    Parameters
    ----------
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    line_density : float
        Number density in longitudinal direction [m^-1].

    Returns
    -------
    float
        Dimensionless space charge perveance
    """
    classical_proton_radius = 1.53469e-18  # [m]
    gamma = 1 + (kin_energy / mass)  # Lorentz factor
    beta = np.sqrt(1.0 - (1.0 / gamma) ** 2)  # velocity/speed_of_light
    return (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)


def env_matrix_to_vector(self, P):
    """Return list of envelope parameters from envelope matrix."""
    return P.ravel()


class DanilovEnvelope22:
    """Class for the beam envelope of the Danilov distribution.

    Attributes
    ----------
    params : ndarray, shape (8,)
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    eps_l : float
        The nonzero rms intrinsic emittance of the beam (eps_1 or eps_2) [m*rad].
    mode : int
        Whether to choose eps_2 = 0 (mode 1) or eps_1 = 0 (mode 2). This amounts
        to choosing the sign of the transverse angular momentum.
    eps_x_frac : float
        ex = eps_x_frac * eps
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    intensity : float
        Number of particles in the bunch represented by the envelope.
    length : float
        Bunch length [m].
    perveance : float
        Dimensionless beam perveance.
    """

    def __init__(
        self,
        eps_l=1.0,
        mode=1,
        eps_x_frac=0.5,
        mass=consts.mass_proton,
        kin_energy=1.0,
        length=1.0,
        intensity=0.0,
        params=None,
    ):
        self.eps_l = eps_l
        self.mode = mode
        self.eps_x_frac = eps_x_frac
        self.eps_y_frac = 1.0 - eps_x_frac
        self.mass = mass
        self.kin_energy = kin_energy
        self.length = length
        self.set_intensity(intensity)
        if params is None:
            eps_x = eps_x_frac * eps_l
            eps_y = (1.0 - eps_x_frac) * eps_l
            rx = 2.0 * np.sqrt(eps_x)
            ry = 2.0 * np.sqrt(eps_y)
            if mode == 1:
                self.params = np.array([rx, 0, 0, rx, 0, -ry, +ry, 0])
            elif mode == 2:
                self.params = np.array([rx, 0, 0, rx, 0, +ry, -ry, 0])
        else:
            self.params = np.array(params)
            eps_x, eps_y = self.apparent_emittances()
            self.eps_l = eps_x + eps_y
            self.eps_x_frac = eps_x / self.eps_l

        # Define bounds on 4D Twiss parameters.
        pad = 1.00e-05
        self.twiss_4d_lb = [
            -np.inf,  # alpha_lx
            +pad,  # beta_lx
            -np.inf,  # alpha_ly
            +pad,  # beta_ly
            +pad,  # u
            +pad,  # nu
        ]
        self.twiss_4d_ub = [
            +np.inf,  # alpha_lx
            +np.inf,  # beta_lx
            +np.inf,  # alpha_ly
            +np.inf,  # beta_ly
            +1.0 - pad,  # u
            +np.pi - pad,  # nu
        ]

    def copy(self):
        """Produced a deep copy of the envelope."""
        return copy.deepcopy(self)

    def set_intensity(self, intensity):
        """Set beam intensity and re-calculate perveance."""
        self.intensity = intensity
        self.line_density = intensity / self.length
        self.perveance = get_perveance(self.mass, self.kin_energy, self.line_density)

    def set_length(self, length):
        """Set beam length and re-calculate perveance."""
        self.length = length
        self.set_intensity(self.intensity)

    def param_vector(self, dim=None):
        """Return list of envelope parameters [a, b, a', b', e, f, e', f']."""
        if dim is None:
            return self.params
        dims = ["x", "xp", "y", "yp"]
        if type(dim) is str:
            dim = dims.index(dim)
        return self.param_matrix()[dim, :]

    def param_matrix(self):
        """Create the envelope matrix P from the envelope parameters.

        The matrix is defined by x = P.c, where x = [x, x', y, y']^T,
        c = [cos(psi), sin(psi)], and '.' means matrix multiplication, with
        0 <= psi <= 2pi. This is useful because any transformation to the
        particle coordinate vector x also done to P. For example, if x -> M x,
        then P -> M P.
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])

    def transform(self, M):
        """Apply matrix M to the coordinates."""
        self.params = np.matmul(M, self.param_matrix()).ravel()

    def rotate(self, phi):
        """Apply clockwise rotation by phi degrees in x-y space."""
        self.transform(get_rotation_matrix_4x4(np.radians(phi)))

    def normalization_matrix(self, kind="2D", inverse=False):
        if kind not in ["2D", "4D"]:
            raise ValueError("Invalid kind")
        if kind == "2D":
            V = CS.normalization_matrix_from_twiss(*self.twiss_2d())
        elif kind == "4D":
            alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu = self.twiss_4d()
            V = LB.normalization_matrix_from_twiss_one_mode(
                alpha_lx=alpha_lx,
                beta_lx=beta_lx,
                alpha_ly=alpha_ly,
                beta_ly=beta_ly,
                u=u,
                nu=nu,
                mode=self.mode,
            )
        if inverse:
            return np.linalg.inv(V)
        return V

    def normalize(self, method="2D", unit_variance=False):
        """Normalize the phase space coordinates.

        Parameters
        ----------
        method : {"2D", "4D"}
            - "2D": The x-x' and y-y' ellipses will be circles of radius sqrt(eps_x)
                    and sqrt(eps_y), where eps_x and eps_y are the rms apparent
                    emittances.
            - "4D": The 4 x 4 covariance matrix becomes diagonal. The x-x' and y-y'
                    ellipses wil be circles of radius radius sqrt(eps_1) and
                    sqrt(eps_2), where eps_1, and eps_2 are the rms intrinsic
                    emittances.
        unit_variance : bool
            Whether to divide by the emittances to scale all coordinates to unit
            variance. This converts all circles to unit circles.

        Returns
        -------
        ndarray, shape (8,)
            The new envelope parameters.
        """
        if method == "2D":
            self.transform(self.normalization_matrix(kind="2D", inverse=True))
            if unit_variance:
                eps_x, eps_y = self.apparent_emittances()
                if eps_x > 0.0:
                    self.params[:4] /= np.sqrt(4.0 * eps_x)
                if eps_y > 0.0:
                    self.params[4:] /= np.sqrt(4.0 * eps_y)
            return self.params
        elif method == "4D":
            r_n = np.sqrt(4.0 * self.eps_l)
            if self.mode == 1:
                self.params = np.array([r_n, 0, 0, r_n, 0, 0, 0, 0])
            elif self.mode == 2:
                self.params = np.array([0, 0, 0, 0, 0, r_n, r_n, 0])
            if unit_variance:
                self.params = self.params / r_n
            return self.params
        else:
            raise ValueError("Invalid method")

    def normalized_param_vector(self, method="2D", unit_variance=False):
        """Return the normalized envelope parameters without changing the envelope parameters.

        Key word arguments are passed to `normalize`.
        """
        params = np.copy(self.params)
        normalized_params = self.normalize(method=method, unit_variance=unit_variance)
        self.params = params
        return normalized_params

    def advance_phases(self, mux=0.0, muy=0.0):
        """Advance the x/y phases.

        It is equivalent to tracking through an uncoupled lattice that the
        envelope is matched to.
        """
        mux = np.radians(mux)
        muy = np.radians(muy)
        V = self.normalization_matrix(kind="2D")
        self.transform(
            np.linalg.multi_dot([V, phase_advance_matrix(mux, muy), np.linalg.inv(V)])
        )

    def projected_tilt_angle(self, x1="x", x2="y"):
        """Return ccw angle of ellipse in x1-x2 plane."""
        a, b = self.param_vector(dim=x1)
        e, f = self.param_vector(dim=x2)
        return 0.5 * np.arctan2(2 * (a * e + b * f), a**2 + b**2 - e**2 - f**2)

    def projected_radii(self, x1="x", x2="y"):
        """Return semi-major and semi-minor axes of ellipse in x1-x2 plane."""
        a, b = self.param_vector(dim=x1)
        e, f = self.param_vector(dim=x2)
        phi = self.projected_tilt_angle(x1, x2)
        sin, cos = np.sin(phi), np.cos(phi)
        sin2, cos2 = sin**2, cos**2
        xx = a**2 + b**2
        yy = e**2 + f**2
        xy = a * e + b * f
        cx = np.sqrt(abs(xx * cos2 + yy * sin2 - 2 * xy * sin * cos))
        cy = np.sqrt(abs(xx * sin2 + yy * cos2 + 2 * xy * sin * cos))
        return np.array([cx, cy])

    def projected_area(self, x1="x", x2="y"):
        """Return area of ellipse in x1-x2 plane."""
        a, b = self.param_vector(dim=x1)
        e, f = self.param_vector(dim=x2)
        return np.pi * abs(a * f - b * e)

    def phases(self):
        """Return horizontal and vertical phases of a particle whose
        coordinates are [x = a, x' = a', y = e, y' = e'].

        The value returned is between zero and 2pi."""
        a, b, ap, bp, e, f, ep, fp = self.normalized_param_vector(method="2D")
        mux = -np.arctan2(ap, a)
        muy = -np.arctan2(ep, e)
        if mux < 0.0:
            mux += 2.0 * np.pi
        if muy < 0.0:
            muy += 2.0 * np.pi
        return (mux, muy)

    def phase_diff(self):
        """Return the x-y phase difference (nu) of all particles in the beam.

        The value returned is in the range [0, pi]. This can also be found from
        the equation cos(nu) = r, where r is the x-y correlation coefficient.
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        if nu < np.pi:
            return nu
        else:
            return 2.0 * np.pi - nu

    def cov(self):
        """Return 4 x 4 transverse covariance matrix."""
        P = self.param_matrix()
        return 0.25 * np.matmul(P, P.T)

    def corr(self):
        """Return 4 x 4 correlation matrix."""
        Sigma = self.cov()
        D = np.sqrt(np.diag(Sigma.diagonal()))
        Dinv = np.linalg.inv(D)
        return np.linalg.multi_dot([Dinv, Sigma, Dinv])

    def apparent_emittances(self):
        """Return rms apparent emittances eps_x, eps_y [m * rad]."""
        Sigma = self.cov()
        eps_x = eps_y = 0.0
        determinant = np.linalg.det(Sigma[:2, :2])
        if determinant > 0.0:
            eps_x = np.sqrt(determinant)
        determinant = np.linalg.det(Sigma[2:, 2:])
        if determinant > 0.0:
            eps_y = np.sqrt(determinant)
        return np.array([eps_x, eps_y])

    def intrinsic_emittances(self, mm_mrad=False):
        """Return rms intrinsic emittances eps1, eps2 [m * rad]."""
        if self.mode == 1:
            return np.array([self.eps_l, 0.0])
        elif self.mode == 2:
            return np.array([0.0, self.eps_l])

    def twiss_2d(self):
        """Return 2D Twiss parameters.

        Order is [alpha_x, beta_x, alpha_y, beta_y].
        """
        Sigma = self.cov()
        eps_x = 0.0
        eps_y = 0.0
        beta_x = np.inf
        beta_y = np.inf
        alpha_x = np.inf
        alpha_y = np.inf
        determinant = np.linalg.det(Sigma[:2, :2])
        if determinant > 0.0:
            eps_x = np.sqrt(determinant)
            beta_x = Sigma[0, 0] / eps_x
            alpha_x = -Sigma[0, 1] / eps_x
        determinant = np.linalg.det(Sigma[2:, 2:])
        if determinant > 0.0:
            eps_y = np.sqrt(determinant)
            beta_y = Sigma[2, 2] / eps_y
            alpha_y = -Sigma[2, 3] / eps_y
        return np.array([alpha_x, beta_x, alpha_y, beta_y])

    def set_twiss_2d(
        self, alpha_x=0.0, beta_x=1.0, alpha_y=0.0, beta_y=1.0, eps_x_frac=None
    ):
        """Set the 2D Twiss parameters of the envelope."""
        V = CS.normalization_matrix_from_twiss(alpha_x, beta_x, alpha_y, beta_y)
        if not eps_x_frac:
            eps_x_frac = self.eps_x_frac
        eps_x = self.eps_l * eps_x_frac
        eps_y = self.eps_l * (1.0 - eps_x_frac)
        A = np.sqrt(4.0 * np.diag([eps_x, eps_x, eps_y, eps_y]))
        self.normalize(method="2D", unit_variance=True)
        self.transform(np.matmul(V, A))

    def twiss_4d(self):
        """Return 4D Twiss parameters as defined by Lebedev/Bogacz.

        Order is [alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu].
        """
        Sigma = self.cov()
        beta_lx = Sigma[0, 0] / self.eps_l
        beta_ly = Sigma[2, 2] / self.eps_l
        alpha_lx = -Sigma[0, 1] / self.eps_l
        alpha_ly = -Sigma[2, 3] / self.eps_l
        if self.mode == 1:
            determinant = np.linalg.det(Sigma[2:, 2:])
            eps_y = np.sqrt(determinant) if determinant > 0.0 else 0.0
            u = eps_y / self.eps_l
        elif self.mode == 2:
            determinant = np.linalg.det(Sigma[:2, :2])
            eps_x = np.sqrt(determinant) if determinant > 0.0 else 0.0
            u = eps_x / self.eps_l
        nu = self.phase_diff()
        return np.array([alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu])

    def set_twiss_4d(
        self, alpha_lx=None, beta_lx=None, alpha_ly=None, beta_ly=None, u=None, nu=None
    ):
        """Set one or more 4D Twiss parameters.

        Parameters set to None are not changed.

        Parameters
        ----------
        - alpha_lx : Horizontal alpha function -<xx'> / eps_l.
        - beta_lx : Horizontal beta function <xx> / eps_l.
        - alpha_ly : Vertical alpha_function -<yy'> / eps_l.
        - beta_ly : Vertical beta function -<yy> / eps_l.
        - u : Coupling parameter in range [0, 1] (eps_y / epsl if mode=1, eps_x / epsl if mode=2).
        - nu : The x-y phase difference in range [0, pi].
        """
        twiss_params = self.twiss_4d()
        for i, value in enumerate([alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu]):
            if value is not None:
                twiss_params[i] = value
        self.set_twiss_4d_vector(twiss_params)

    def set_twiss_4d_vector(self, twiss_params):
        """Set the 4D Twiss parameters of the envelope.

        Parameters
        ----------
        `twiss_params` : list, shape (6,)
            Array containing the 4D Twiss params for a single mode:
            - alpha_lx : Horizontal alpha function -<xx'> / epsl.
            - beta_lx : Horizontal beta function <xx> / epsl.
            - alpha_ly : Vertical alpha_function -<yy'> / epsl.
            - beta_ly : Vertical beta function -yy> / epsl.
            - u : Coupling parameter in range [0, 1]. This is equal to eps_y / epsl
                  in mode 1 or eps_x / epsl in mode 2.
            - nu : The x-y phase difference in range [0, pi].
        """
        alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu = twiss_params
        V = LB.normalization_matrix_from_twiss_one_mode(
            alpha_lx=alpha_lx,
            beta_lx=beta_lx,
            alpha_ly=alpha_ly,
            beta_ly=beta_ly,
            u=u,
            nu=nu,
            mode=1,
        )
        self.normalize(method="4D")
        self.transform(V)

    def set_cov(self, Sigma, verbose=0):
        """Set the beam covariance matrix `Sigma`."""

        def residuals(params, Sigma):
            self.params = params
            return 1.00e06 * get_moment_vector(Sigma - self.cov())

        result = opt.least_squares(
            residuals, self.params, args=(Sigma,), xtol=1.00e-12, verbose=verbose
        )
        return result.x
    
    def get_particle_coordinates(self, psi=0.0):
        """Return the 4D phase space coordinates of a particle on the envelope.

        psi is in the range [0, 2pi].
        """
        return np.matmul(self.param_matrix(), [np.cos(psi), np.sin(psi)])

    def generate_dist(self, nparts=1, density="uniform"):
        """Generate a distribution of particles from the envelope.

        Parameters
        ----------
        nparts : int
            The number of simulation particles in the distribution.
        density : {'uniform', 'on_ellipse', 'gaussian'}
            How to fill the distribution.
                'uniform': fill envelope with uniform density
                'on_ellipse': generate particles only on the envelope
                'gaussian': fill envelope with Gaussian density

        Returns
        -------
        ndarray, shape (nparts, 4)
            The coordinate array for the distribution.
        """
        nparts = int(nparts)
        psis = np.linspace(0, 2 * np.pi, nparts)
        X = np.array([self.get_particle_coordinates(psi) for psi in psis])
        if density == "uniform":
            radii = np.sqrt(np.random.random(nparts))
        elif density == "on_ellipse":
            radii = np.ones(nparts)
        elif density == "gaussian":
            radii = np.random.normal(size=nparts)
        return radii[:, np.newaxis] * X

    def from_bunch(self, bunch):
        """Extract the envelope parameters from a Bunch object."""
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params

    def to_bunch(self, nparts=0, no_env=False):
        """Add the envelope parameters to a Bunch object. The first two
        particles represent the envelope parameters.

        Parameters
        ----------
        nparts : int
            The number of particles in the bunch. The bunch will just hold the
            envelope parameters if nparts == 0.
        no_env : bool
            If True, do not include the envelope parameters in the first
            two bunch particles.

        Returns
        -------
        bunch: Bunch object
            The bunch representing the distribution of size 2 + nparts
            (unless `no_env` is True).
        params_dict : dict
            The dictionary of parameters for the bunch.
        """
        bunch, params_dict = initialize_bunch(self.mass, self.kin_energy)
        if not no_env:
            a, b, ap, bp, e, f, ep, fp = self.params
            bunch.addParticle(a, ap, e, ep, 0.0, 0.0)
            bunch.addParticle(b, bp, f, fp, 0.0, 0.0)
        for x, xp, y, yp in self.generate_dist(nparts):
            z = np.random.uniform(0, self.length)
            bunch.addParticle(x, xp, y, yp, z, 0.0)
        if nparts > 0:
            bunch.macroSize(self.intensity / nparts if self.intensity > 0.0 else 1.0)
        return bunch, params_dict

    def track(self, lattice, nturns=1, ntestparts=0, progbar=False):
        """Track the envelope through the lattice.

        The envelope parameters are updated after it is tracked. If
        `ntestparts` is nonzero, test particles will be tracked which receive
        linear space charge kicks based on the envelope parameters.
        """
        bunch, params_dict = self.to_bunch(ntestparts)
        turns = trange(nturns) if progbar else range(nturns)
        for _ in turns:
            lattice.trackBunch(bunch, params_dict)
        self.from_bunch(bunch)

    def track_store_params(self, lattice, nturns=1):
        """Track and return the turn-by-turn envelope parameters."""
        params_tbt = [self.params]
        for _ in range(nturns):
            self.track(lattice)
            params_tbt.append(self.params)
        return params_tbt

    def get_tunes(self, lattice):
        """Return the fractional horizontal and vertical tunes."""
        env = self.copy()
        mux0, muy0 = env.phases()
        env.track(lattice)
        mux1, muy1 = env.phases()
        tune_x = ((mux1 - mux0) / (2 * np.pi)) % 1
        tune_y = ((muy1 - muy0) / (2 * np.pi)) % 1
        return np.array([tune_x, tune_y])

    def get_transfer_matrix(self, lattice):
        """Compute the linear transfer matrix with space charge included.

        The method is taken from /src/teapot/MatrixGenerator.cc. That method
        computes the 7x7 transfer matrix, but we just need the 4x4 matrix.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice may have envelope solver nodes. These nodes should
            track the beam envelope using the first two particles in the bunch,
            then use these to apply the appropriate linear space charge kicks
            to the rest of the particles.

        Returns
        -------
        M : ndarray, shape (4, 4)
            The 4x4 linear transfer matrix of the combined lattice + space
            charge focusing system.
        """
        # If space charge is zero, we can just use the TEAPOT_MATRIX_Lattice
        # class to calculate the transfer matrix.
        if self.perveance == 0:
            M = get_transfer_matrix(lattice, self.mass, self.kin_energy)
            return M[:4, :4]

        # The envelope parameters will change if the beam is not matched to
        # the lattice, so make a copy of the intial state.
        env = self.copy()

        step_arr_init = np.full(6, 1.0e-6)
        step_arr = np.copy(step_arr_init)
        step_reduce = 20.0
        bunch, params_dict = env.to_bunch()
        bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(step_arr[0] / step_reduce, 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, step_arr[1] / step_reduce, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, step_arr[2] / step_reduce, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, 0.0, step_arr[3] / step_reduce, 0.0, 0.0)
        bunch.addParticle(step_arr[0], 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, step_arr[1], 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, step_arr[2], 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, 0.0, step_arr[3], 0.0, 0.0)

        lattice.trackBunch(bunch, params_dict)
        X = [
            [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
            for i in range(2, bunch.getSize())
        ]
        X = np.array(X)
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                x1 = step_arr[i] / step_reduce
                x2 = step_arr[i]
                y0 = X[0, j]
                y1 = X[i + 1, j]
                y2 = X[i + 1 + 4, j]
                M[j, i] = ((y1 - y0) * x2 * x2 - (y2 - y0) * x1 * x1) / (
                    x1 * x2 * (x2 - x1)
                )
        return M

    def match_bare(self, lattice, method="auto", solver_nodes=None):
        """Match to the lattice without space charge.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice in which to match. If envelope solver nodes nodes are
            in the lattice, a list of these nodes needs to be passed as the
            `solver_nodes` parameter so they can be turned off/on.
        method : str
            If '4D', match to the lattice using the eigenvectors of the
            transfer matrix. This may result in the beam being completely
            flat, for example when the lattice is uncoupled. The '2D' method
            will only match the x-x' and y-y' ellipses of the beam.
        solver_nodes : list, optional
            List of nodes which are sublasses of SC_Base_AccNode. If provided,
            all space charge nodes are turned off, then the envelope is matched
            to the bare lattice, then all space charge nodes are turned on.

        Returns
        -------
        ndarray, shape (8,)
            The matched envelope parameters.
        """
        if solver_nodes is not None:
            for node in solver_nodes:
                node.switcher = False

        # Get linear transfer matrix
        M = self.get_transfer_matrix(lattice)
        tmat = TransferMatrix(M)
        if not tmat.stable:
            print("WARNING: transfer matrix is not stable.")

        # Match to the bare lattice.
        if method == "auto":
            method = "4D" if tmat.coupled else "2D"
        if method == "2D":
            tmat = TransferMatrixCourantSnyder(M)
            tmat.analyze()
            self.set_twiss_2d(
                alpha_x=tmat.params["alpha_x"],
                beta_x=tmat.params["beta_x"],
                alpha_y=tmat.params["alpha_y"],
                beta_y=tmat.params["beta_y"],
            )
        elif method == "4D":
            tmat = TransferMatrixCourantSnyder(M)
            tmat.analyze()
            V = tmat.params["V"]
            self.normalize(method="4D")
            self.transform(V)

        # If rms beam size or divergence are zero in either plane, make them
        # slightly nonzero. This could occur when the lattice is uncoupled and
        # has unequal tunes.
        a, b, ap, bp, e, f, ep, fp = self.params
        pad = 1.00e-08
        if np.all(np.abs(self.params[:4]) < pad):
            a = bp = pad
        if np.all(np.abs(self.params[4:]) < pad):
            f = ep = pad
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])

        if solver_nodes is not None:
            for node in solver_nodes:
                node.switcher = True

        return self.params

    def match(self, lattice, solver_nodes=None, method="auto", tol=1.00e-04, **kws):
        """Match the envelope to the lattice.

        lattice : TEAPOT_Lattice
            The lattice to match to.
        solver_nodes : list[DanilovEnvSolver]
            A list of envelope solver nodes.
        method : {'auto', 'lsq', 'replace_avg'}
            The matching method to use (defined in the functions below). If
            'auto', the 'lsq' method will be tried first. If the final cost
            function is above `tol`, then it will try the 'replace_avg' method.
            Usually 'lsq' will work just fine, but I found that a lattice with
            skew quadrupoles can result in a matched beam which has an x-y
            correlation coefficient of +1 or -1 (nu = 0 or pi). The 'lsq'
            method can fail in this case. Details are found in [1] (see the
            top of this module).
        tol : float
            If the final cost function of the 'lsq' method is above `tol`, the
            the 'replace_avg' method will be tried.
        **kws
            Key word arguments for the matching method.
        """
        if self.perveance == 0:
            return self.match_bare(lattice, "auto", solver_nodes)

        def initialize():
            self.set_twiss_4d(u=0.5)
            self.set_twiss_4d(nu=(0.5 * np.pi))
            self.match_bare(lattice, "2D", solver_nodes)

        initialize()

        if method == "lsq":
            return self._match_lsq(lattice, **kws)
        elif method == "replace_avg":
            return self._match_replace_avg(lattice, **kws)
        elif method == "auto":
            result = self._match_lsq(lattice, **kws)
            if result.cost > tol:
                print("Cost = {:.2e} > tol.".format(result.cost))
                print("Trying 'replace by average' method.")
                initialize()
                result = self._match_replace_avg(lattice, **kws)
            return result
        else:
            raise ValueError("Invalid method! Options: {'lsq', 'replace_avg', 'auto'}")

    def _residuals(self, lattice, factor=1.0e6):
        """Return initial minus final beam moments after tracking.
        The method does not change the envelope.
        """
        env = self.copy()
        Sigma0 = env.cov()
        env.track(lattice)
        Sigma1 = env.cov()
        return factor * get_moment_vector(Sigma1 - Sigma0)

    def _match_lsq(self, lattice, **kws):
        """Compute matched envelope using scipy.least_squares optimizer.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be in
            place.
        **kws
            Keyword arguments to be passed to `scipy.optimize.least_squares`
            method.

        Returns
        -------
        result : scipy.optimize.OptimizeResult object
            See the documentation for the description. The two important
            fields are `x`: the final parameter vector, and `cost`: the final
            cost function.
        """

        def cost_func(twiss_params):
            self.set_twiss_4d_vector(twiss_params)
            return self._residuals(lattice)

        print(self.twiss_4d_lb, self.twiss_4d_ub)
        result = opt.least_squares(
            cost_func,
            self.twiss_4d(),
            bounds=(self.twiss_4d_lb, self.twiss_4d_ub),
            **kws
        )
        self.set_twiss_4d_vector(result.x)
        return result

    def _match_replace_avg(
        self,
        lattice,
        nturns_avg=15,
        max_iters=100,
        tol=1.0e-6,
        ftol=1.0e-8,
        xtol=1.0e-8,
        verbose=0,
    ):
        """Compute the matched envelope using custom optimizer.

        The method works be tracking the beam for a number of turns, then
        computing the average of the oscillations of the 4D twiss parameters.
        This average is used to generate the beam for the next iteration, and
        this is repeated until convergence.

        NOTE: it is not gauranteed to converge!

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be in
            place.
        nturns_avg : int
            Number of turns to average over when updating the parameter vector.
        max_iters : int
            Maximum number of iterations to perform.
        tol : float
            Tolerance for the value of the cost function.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the parameter vector
            norm. The parameter vector is just the six-element vector of the
            4D Twiss parameters.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
                * 0 : work silently (default)
                * 1 : display a termination report
                * 2 : display progress during iterations
        """

        def get_avg_p():
            p_tracked = []
            for _ in range(nturns_avg + 1):
                p_tracked.append(self.twiss_4d())
                self.track(lattice)
            return np.mean(p_tracked, axis=0)

        def is_converged(cost, cost_reduction, step_norm, p):
            converged, message = False, "Did not converge."
            if cost < tol:
                converged == True
                msg = "`tol` termination condition is satisfied."
            if abs(cost_reduction) < ftol * cost:
                converged = True
                msg = "`ftol` termination condition is satisfied."
            if abs(step_norm) < xtol * (xtol + np.linalg.norm(p)):
                converged = True
                message = "`xtol` termination condition is satisfied."
            return converged, message

        if self.perveance == 0:
            return self.match_bare(lattice, method="auto")

        iteration = 0
        old_p = self.twiss_4d()
        old_cost = np.inf
        history = [old_p]
        converged = False
        message = "Did not converge."
        t_start = time.time()
        if verbose == 2:
            print_header()
        while not converged and iteration < max_iters:
            iteration += 1
            p = get_avg_p()
            self.set_twiss_4d_vector(p)
            cost = np.sum(self._residuals(lattice) ** 2)
            cost_reduction = old_cost - cost
            step_norm = np.linalg.norm(p - old_p)
            converged, message = is_converged(cost, cost_reduction, step_norm, p)
            old_p, old_cost = p, cost
            history.append(p)
            if verbose == 2:
                print_iteration(iteration, cost, cost_reduction, step_norm)
        t_end = time.time()
        if verbose > 0:
            print("   ".format(message))
            print("   cost = {:.4e}".format(cost))
            print("   iters = {}".format(iteration))
        return MatchingResult(p, cost, iteration, t_end - t_start, message, history)

    def perturb_twiss(self, radius=0.1):
        """Randomly perturb the 4D Twiss parameters."""
        lo = 1.0 - radius
        hi = 1.0 + radius
        twiss_params = self.twiss_4d()
        twiss_params = np.random.uniform(lo * twiss_params, hi * twiss_params)
        twiss_params = np.clip(twiss_params, self.twiss_4d_lb, self.twiss_4d_ub)
        self.set_twiss_4d_vector(twiss_params)

    def print_twiss_2d(self, indent=4):
        alpha_x, beta_x, alpha_y, beta_y = self.twiss_2d()
        eps_x, eps_y = self.apparent_emittances()
        print("2D Twiss parameters:")
        print("  alpha_x = {:}".format(alpha_x))
        print("  alpha_y = {:}".format(alpha_y))
        print("  beta_x = {:} [m/rad]".format(beta_x))
        print("  beta_y = {:} [m/rad]".format(beta_y))
        print("  eps_x = {:} [mm*mrad]".format(1.00e06 * eps_x))
        print("  eps_y = {:} [mm*mrad]".format(1.00e06 * eps_y))

    def print_twiss_4d(self):
        alpha_lx, beta_lx, alpha_ly, beta_ly, u, nu = self.twiss_4d()
        eps_1, eps_2 = self.intrinsic_emittances()
        print("4D Twiss parameters:")
        print("  mode (l) = {}".format(self.mode))
        print("  alpha_lx = {:}".format(alpha_lx))
        print("  alpha_ly = {:}".format(alpha_ly))
        print("  beta_lx = {:} [m/rad]".format(beta_lx))
        print("  beta_ly = {:} [m/rad]".format(beta_ly))
        print("  u = {:}".format(u))
        print("  nu = {:} [deg]".format(np.degrees(nu)))
        print("  eps_1 = {:} [mm*mrad]".format(1.00e06 * eps_1))
        print("  eps_2 = {:} [mm*mrad]".format(1.00e06 * eps_2))


class MatchingResult:
    """Class to store the results of custom matching algorithm"""

    def __init__(self, p, cost, iters, runtime, message, history):
        self.p, self.cost, self.iters, self.time = p, cost, iters, runtime
        self.message = message
        self.history = np.array(history)


def print_header():
    print(
        "{0:^15}{1:^15}{2:^15}{3:^15}".format(
            "Iteration", "Cost", "Cost reduction", "Step norm"
        )
    )


def print_iteration(iteration, cost, cost_reduction, step_norm):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.3e}".format(cost_reduction)

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    print("{0:^15}{1:^15.4e}{2}{3}".format(iteration, cost, cost_reduction, step_norm))
