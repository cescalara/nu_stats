import numpy as np
from astropy import units as u

from .cosmology import luminosity_distance
from .distributions import BoundedPowerLaw


class PowerLaw:
    """
    Simple power law spectrum.
    """

    #@u.quantity_input # Doesn't support kwargs (*)
    def __init__(
        self,
        gamma: float,
        Emin: u.GeV,
        Emax: u.GeV,
        Enorm: u.GeV,
        *,
        Flnorm: 1 / (u.GeV * u.cm ** 2 * u.s) = np.nan/ (u.GeV * u.cm ** 2 * u.s),
        L: u.erg / u.s = np.nan * u.erg / u.s,
        z: float = np.nan,
    ):
        """
        params:
            gamma (float): Spectral Index
            Emin (u.GeV): Min energy
            Emax (u.GeV): Max energy
            Enorm (u.GeV): Normalisation energy
            Flnorm (1 / (u.GeV * u.cm ** 2 * u.s), optional): flux at enorm. Defaults to nan, for which it is retrieved through L
            L (u.erg, optional): Source luminosity. Defaults to nan, for which above is used
            z (float, optional): Source redshift. Defaults to np.nan.
        """
        assert np.isnan(L) != np.isnan(Flnorm), 'Pass either Flnorm or L, not both'
        assert np.isnan(L) == np.isnan(z), 'Pass the source redshift'

        # (*) Manual quantity check
        assert isinstance(gamma, float)
        assert Emin.unit == 'GeV'
        assert Emin.unit == 'GeV'
        assert Enorm.unit == 'GeV'
        assert Flnorm.unit == '1 / (cm2 GeV s)'
        assert L.unit == 'erg / s'
        assert isinstance(z, float)

        self.gamma = gamma
        self.Emin = Emin
        self.Emax = Emax
        self.Enorm = Enorm
        self.L = L
        self.z = z
        if np.isnan(Flnorm): 
            #  Compute some useful quantities
            self.D = luminosity_distance(self.z)
            self.F = self.L / (4 * np.pi * self.D ** 2)
            self.F = self.F.to(u.GeV / (u.cm ** 2 * u.s))
            self.Flnorm = self._get_norm()
        else:
            self.Flnorm = Flnorm

        self.power_law_model = BoundedPowerLaw(
            self.gamma, self.Emin.value, self.Emax.value
        )

    @u.quantity_input
    def spectrum(self, E: u.GeV):
        """
        dN/dEdAdt
        """

        return self.Flnorm * np.power(E, -self.gamma)

    @u.quantity_input
    def integrate(self, Emin, Emax):
        """
        Integrate the power law between Emin and Emax.
        """

        int_norm = self.Flnorm / (np.power(self.Enorm, -self.gamma) * (1 - self.gamma))

        return int_norm * (
            np.power(Emax, 1 - self.gamma) - np.power(Emin, 1 - self.gamma)
        )

    def sample(self, N=1):
        """
        Sample N energies from this distribution.
        """
        if N == 1:

            return self.power_law_model.samples(N)[0] * u.GeV

        else:

            return self.power_law_model.samples(N) * u.GeV
    
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
