import numpy as np
from astropy import units as u

from .distributions import BoundedPowerLaw


class PowerLaw:
    """
    Simple power law spectrum.
    """

    @u.quantity_input
    def __init__(
        self,
        norm: 1 / (u.GeV * u.cm ** 2 * u.s),
        Emin: u.GeV,
        Emax: u.GeV,
        Enorm: u.GeV,
        gamma: float,
    ):

        self.norm = norm
        self.Emin = Emin
        self.Emax = Emax
        self.Enorm = Enorm
        self.gamma = gamma
        self.power_law_model = BoundedPowerLaw(
            self.gamma, self.Emin.value, self.Emax.value
        )

    @u.quantity_input
    def spectrum(self, E: u.GeV):
        """
        dN/dEdAdt
        """

        return self.norm * np.power(E, -self.gamma)

    @u.quantity_input
    def integrate(self, Emin: u.GeV, Emax: u.GeV):
        """
        Integrate the power law between Emin and Emax.
        """

        int_norm = self.norm / (np.power(self.Enorm, -self.gamma) * (1 - self.gamma))

        return int_norm * (
            np.power(self.Emax, 1 - self.gamma) - np.power(self.Emin, 1 - self.gamma)
        )

    def sample(self, N=1):
        """
        Sample N energies from this distribution.
        """
        if N == 1:

            return self.power_law_model.samples(N)[0] * u.GeV

        else:

            return self.power_law_model.samples(N) * u.GeV
