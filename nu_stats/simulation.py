import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot

from astropy import units as u
from astropy.coordinates import SkyCoord
import collections

from vMF.vMF import sample_vMF

from .cosmology import luminosity_distance
from .spectra import PowerLaw
from .plotting import SphericalCircle


class Simulation:
    """
    Setting up and running simulations.
    """

    @u.quantity_input
    def __init__(
        self,
        L: u.erg / u.s,
        gamma: float,
        z: float,
        F_diff: 1 / (u.GeV * u.cm ** 2 * u.s),
        Emin: u.GeV = 1e5,
        Emax: u.GeV = 1e8,
        Enorm: u.GeV = 1e5,
    ):
        """
        :param L: Luminosity of source
        :param gamma: Spectral index of source
        :param z: Redshift of source
        :param F_diff: Diffuse backround flux at Emin
        :param Emin: Minimum energy
        :param Emax: Maximum energy
        :param Enorm: Normalisation energy
        """

        self.L = L
        self.gamma = gamma
        self.z = z
        self.F_diff = F_diff
        self.Emin = Emin
        self.Emax = Emax
        self.Enorm = Enorm

        #  Compute some useful quantities
        self.D = luminosity_distance(self.z)
        self.F = self.L / (4 * np.pi * self.D ** 2)
        self.F = self.F.to(u.GeV / (u.cm ** 2 * u.s))
        self.f = self._get_associated_fraction()

        # Make power law spectra
        ps_norm = self._get_norm()
        self.point_source = PowerLaw(
            ps_norm, self.Emin, self.Emax, self.Enorm, self.gamma
        )
        self.diffuse_bg = PowerLaw(
            self.F_diff, self.Emin, self.Emax, self.Enorm, self.gamma
        )
        self.z_bg = 1  # Assume bg at redshift 1

        # Assume simple constant effective area for now
        self.effective_area = 1 * u.m ** 2
        self.ang_reco_err = 6 * u.deg

        # Assume fixed observation time
        self.time = 10 * u.yr

        # Assume fixed source position
        self.ra = 180 * u.deg
        self.dec = 5 * u.deg
        self.coord = SkyCoord(self.ra, self.dec, frame="icrs")

        # Store truth for comparison with fits
        self.truth = collections.OrderedDict()
        self.truth["L"] = self.L
        self.truth["gamma"] = self.gamma
        self.truth["F_diff"] = self.F_diff

    def run(self, seed=42):
        """
        Run simulation.
        """

        # Set seed
        np.random.seed(seed)

        Nex, weights = self._get_N_expected()

        N = np.random.poisson(Nex)
        print("Simulating %i events..." % N)

        # Get source direction as a unit vector
        self.coord.representation_type = "cartesian"
        source_dir = np.array(
            [self.coord.x.value, self.coord.y.value, self.coord.z.value]
        )
        self.coord.representation_type = "spherical"

        # Convert angular error to vMF kappa
        kappa = 7552 * np.power(self.ang_reco_err.value, -2)

        # Sample labels 0 <=> PS, 1 <=> BG
        self.labels = np.random.choice([0, 1], p=weights, size=N)

        self.Etrue = np.zeros_like(self.labels) * u.GeV
        self.Earr = np.zeros_like(self.labels) * u.GeV
        self.Edet = np.zeros_like(self.labels) * u.GeV
        self.true_dir = np.zeros((N, 3))
        self.det_dir = np.zeros((N, 3))

        for i in range(N):

            if self.labels[i] == 0:

                self.Etrue[i] = self.point_source.sample()
                self.Earr[i] = self.Etrue[i] / (1 + self.z)
                self.true_dir[i] = source_dir

            elif self.labels[i] == 1:

                self.Etrue[i] = self.diffuse_bg.sample()
                self.Earr[i] = self.Etrue[i] / (1 + self.z_bg)
                self.true_dir[i] = sample_vMF(np.array([1, 0, 0]), 0, 1)

            self.Edet[i] = np.random.lognormal(np.log(self.Earr[i].value), 0.5) * u.GeV

            self.det_dir[i] = sample_vMF(self.true_dir[i], kappa, 1)

        # Convert unit vectors to coords
        self.det_coord = SkyCoord(
            self.det_dir.T[0],
            self.det_dir.T[1],
            self.det_dir.T[2],
            representation_type="cartesian",
            frame="icrs",
        )
        self.det_coord.representation_type = "spherical"

        print("Done!")

    def show_spectrum(self):
        """
        Show simulated spectrum.
        """

        bins = 10 ** np.linspace(
            np.log10(self.Emin.value) - 1, np.log10(self.Emax.value)
        )

        fig, ax = plt.subplots()
        ax.hist(self.Etrue.value, bins=bins, alpha=0.7, label="Etrue")
        ax.hist(self.Earr.value, bins=bins, alpha=0.7, label="Earr")
        ax.hist(self.Edet.value, bins=bins, alpha=0.7, label="Edet")
        ax.set_xscale("log")
        ax.legend()

    def show_skymap(self):
        """
        Show simulated directions.
        """

        label_cmap = plt.cm.Set1(list(range(2)))

        fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
        fig.set_size_inches((7, 5))

        for ra, dec, l in zip(
            self.det_coord.icrs.ra,
            self.det_coord.icrs.dec,
            self.labels,
        ):

            circle = SphericalCircle(
                (ra, dec),
                self.ang_reco_err,
                color=label_cmap[l],
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )

            ax.add_patch(circle)

    def _get_N_expected(self):
        """
        Calculate the expected number of events.
        """

        time = self.time.to(u.s)
        aeff = self.effective_area.to(u.cm ** 2)

        # point_source
        ps_int = self.point_source.integrate(self.Emin, self.Emax)
        z_factor = np.power(1 + self.z, -self.gamma)
        Nex_ps = time * aeff * ps_int * z_factor

        # diffuse bg
        bg_int = self.diffuse_bg.integrate(self.Emin, self.Emax)
        z_factor = np.power(1 + self.z_bg, -self.gamma)
        Nex_bg = time * aeff * bg_int * z_factor

        # Weights for sampling
        Nex = Nex_ps.value + Nex_bg.value
        weights = [Nex_ps.value / Nex, Nex_bg.value / Nex]

        return Nex, weights

    def _get_associated_fraction(self):
        """
        Calculate the associated fraction.
        """

        pass

    def _get_norm(self):
        """
        Get power law spectrum normalisation.
        """

        if self.gamma == 2:

            int_norm = 1 / np.power(self.Enorm, -self.gamma)
            power_int = int_norm * np.log(self.Emax / self.Emin)

        else:

            int_norm = 1 / (np.power(self.Enorm, -self.gamma) * (2 - self.gamma))
            power_int = int_norm * (
                np.power(self.Emax, 2 - self.gamma)
                - np.power(self.Emin, 2 - self.gamma)
            )

        return self.F / power_int
