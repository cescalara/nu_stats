import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
import collections

from .cosmology import luminosity_distance
from .spectra import PowerLaw


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

        # Assume simple constant effective area for now
        self.effective_area = 1 * u.m ** 2

        # Assume fixed observation time
        self.time = 10 * u.yr

        # Assume fixed source position
        self.ra = 180 * u.deg
        self.dec = 5 * u.deg

        # Store truth for comparison with fits
        self.truth = collections.OrderedDict()
        self.truth["L"] = self.L
        self.truth["gamma"] = self.gamma
        self.truth["F_diff"] = self.F_diff

    def run(self):
        """
        Run simulation.
        """

        pass

    def show_spectrum(self):
        """
        Show simulated spectrum.
        """

        pass

    def show_skymap(self):
        """
        Show simulated directions.
        """

        pass

    def _get_N_expected(self):
        """
        Calculate the expected number of events.
        """

        pass

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
