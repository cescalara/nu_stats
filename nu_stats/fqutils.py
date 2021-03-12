from collections import OrderedDict
import numpy as np
from astropy import units as u

from nu_stats.simulation import Simulation


class MarginalisedEnergyLikelihood:
    def __init__(
        self,
        energy,
        sim_index=1.5,
        min_index=1.5,
        max_index=4.0,
        min_E: u.GeV = 1e3,
        max_E: u.GeV = 1e10,
        Ebins=50,
    ):
        """
        Compute the marginalised energy likelihood by using a 
        simulation of a large number of reconstructed muon 
        neutrino tracks.
        
        :param energy: Reconstructed muon energies (preferably many).
        :param sim_index: Spectral index of source spectrum in sim.
        """

        self._energy = energy

        self._sim_index = sim_index
        self._min_index = min_index
        self._max_index = max_index

        self._min_E = min_E
        self._max_E = max_E

        self._index_bins = np.linspace(min_index, max_index)

        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Ebins)  # GeV

        self._precompute_histograms()

    def _precompute_histograms(self):
        self._likelihood = np.zeros(
            (len(self._index_bins[:-1]), len(self._energy_bins[:-1]))
        )

        for i, index in enumerate(self._index_bins[:-1]):

            index_center_bin = index + (self._index_bins[i + 1] - index) / 2

            weights = self._calc_weights(index_center_bin)

            hist, _ = np.histogram(
                np.log10(self._energy),
                bins=self._energy_bins,
                weights=weights,
                density=True,
            )

            self._likelihood[i] = hist

    def _calc_weights(self, new_index):
        return np.power(self._energy, self._sim_index - new_index)

    def __call__(self, E, new_index):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        if E < self._min_E or E > self._max_E:
            raise ValueError(
                "Energy "
                + str(E)
                + "is not in the accepted range between "
                + str(self._min_E)
                + " and "
                + str(self._max_E)
            )

        if new_index < self._min_index or new_index > self._max_index:
            raise ValueError(
                "Sepctral index "
                + str(new_index)
                + " is not in the accepted range between "
                + str(self._min_index)
                + " and "
                + str(self._max_index)
            )

        i_index = np.digitize(new_index, self._index_bins) - 2
        E_index = np.digitize(np.log10(E), self._energy_bins) - 2

        return self._likelihood[i_index][E_index]


class FqStructure:
    def __init__(
        self,
        z: float = np.nan,
        Emin: u.GeV = np.nan,
        Emax: u.GeV = np.nan,
        Esim_gamma: float = np.nan,
        n_Esim: int = np.nan,
        E_seed = 123,
    ):
        """
        Class for holding useful stuff to do with the Braun paper approach  

        No Args: Spacial model only, i.e. P(Edet|gamma)=1

        Args: Properties for simulation for precomputing P(Edet|gamma)           
            z (float, optional): Defaults to np.nan.
            Emin (u.GeV, optional): Defaults to np.nan.
            Emax (u.GeV, optional): Defaults to np.nan.
            Esim_gamma (float, optional):  Defaults to np.nan.
            n_Esim (int, optional): Defaults to np.nan.
            E_seed (int, optional): Defaults to 123.
        """    
        self.E_input_array = [z, Emin, Emax, Esim_gamma, n_Esim, E_seed]
        empty_entries = np.array([
            np.isnan(entry) for entry in self.E_input_array[:-1]
            ])
        self.spacial_only = empty_entries.any()
        if self.spacial_only:
            assert empty_entries.all(), f'''
            Missing arguments for energy sim.
            In positional order, True means missing:
            {empty_entries}
            ''' 

        self._construct_energy_likelihood()
    
    def _construct_energy_likelihood(self):
        if self.spacial_only:
            self.energy_likelihood = lambda Edet, spectral_index: 1
            print('Marginalized energy likelihood taken to be 1.')
        else:
            print('Generating marginalized energy likelihood..', end='\r')
            z, Emin, Emax, Esim_gamma, n_Esim, E_seed = self.E_input_array
            '''
            Because we only want a sample of Edet for a certain gamma with no
            spacial factor, we can use a single source 
            '''
            self.Esim = Simulation(
                1*u.erg/u.s,
                Esim_gamma, 
                z, 
                0/(u.GeV * u.cm**2 * u.s), 
                Emin, 
                Emax, 
                Emin, 
                n_Esim)
            self.Esim.run(seed = E_seed)
            Esim_dat = self.Esim.get_data()
            self.energy_likelihood = MarginalisedEnergyLikelihood(
                Esim_dat['Edet'], Esim_gamma
            )
            print('Marginalized energy likelihood generated.')

    def source_likelihood(self,
        E_r : float, # Event Edet
        obs_dir: np.ndarray, # Event
        gamma: float, # Source
        source_dir: np.ndarray, # Source
        kappa: float # Detector
    ):
        """ Evaluate source likelihood
        Args:
            source_dir (np.ndarray): direction of source
            obs_dir (np.ndarray): direction of observed neutrino
            kappa (float): uncertainty of neutrino direction in obs
        """
        assert source_dir.ndim == 1, \
            f'source_dir.ndim = {source_dir.ndim}, expected 1'
        assert obs_dir.ndim == 1, \
            f'obs_dir.ndim = {obs_dir.ndim}, expected 1'

        spacial_factor = (kappa/np.pi
            * np.exp(-kappa
                    * sqeuclidean(source_dir - obs_dir)/2
                    )
                ) # bivariate normal with kappa = 1/sigma^2
        
        energy_factor = self.energy_likelihood(E_r, gamma) # p(E_r|gamma)
        #print('spacial', spacial_factor)
        #print('energy ', energy_factor)
        return spacial_factor * energy_factor

    def bg_likelihood(self, E_r, gamma):
        """
        If spacial is isotropic, normalisation doesn't matter
        since the TS is relative
        """
        if self.spacial_only:
            return 1
        else:
            return self.energy_likelihood(E_r, gamma) # p(E_r|gamma)

    def test_stat(self,
        E_r : np.ndarray, # Edets from sim
        obs_dir: np.ndarray, # det_dir from sim
        gamma: float,
        source_dir: np.ndarray,
        kappa: float # Detector
    ):
        N_obs = obs_dir.shape[0]
        TS = np.empty(N_obs)
        TS[:] = np.NaN
        for j in range(N_obs):
            S = 0
            for i in range(source_dir.shape[0]):
                S += self.source_likelihood(
                    E_r[j],
                    obs_dir[j],
                    gamma,
                    source_dir,
                    kappa
                    )
            TS[j] = 2*np.log(S / self.bg_likelihood(E_r[j], gamma))
        return TS
   
    def event_statistics(self,
        fit_input: OrderedDict,
        bg_dat: OrderedDict,
        gamma: float
    ):
        """Get p-values for fit_input events coming from bg
        Args:
            fit_input (OrderedDict): output from Simulation.get_data()
            bg_dat (OrderedDict): output from background Simulation.get_data()
            gamma (float): 

        Returns:
            np.ndarray: 
        """    
        sim_TS = self.test_stat(
            fit_input['Edet'],
            fit_input['det_dir'],
            gamma,
            fit_input['source_dir'],
            fit_input['kappa']
            )
        bg_TS = self.test_stat(
            bg_dat['Edet'],
            bg_dat['det_dir'],
            gamma,
            fit_input['source_dir'],
            bg_dat['kappa']
            )

        ## get p values
        sim_p = np.zeros_like(sim_TS)
        for i,obs_TS in enumerate(sim_TS):
            sim_p[i] = (np.mean(bg_TS > obs_TS))
        return sim_TS, bg_TS, sim_p
    
    def log_band_likelihood(self, fit_input, n_s, gamma):
        '''
        Returns log(L(x_s,n_s,gamma)) (7) to be maximized w.r.t. n_s and gamma
        for their estimates.
        i.e. sum_i(log(n_s/N S_i + (1-n_s/N) B_i)
        '''
        f = n_s/fit_input['N']
        log_likelihoods = [np.log(
            f * self.source_likelihood(
                fit_input['Edet'][i],
                fit_input['det_dir'][i],
                gamma,
                fit_input['source_dir'],
                fit_input['kappa']
                )
            + (1-f) * self.bg_likelihood(fit_input['Edet'][i], gamma)
            )
            for i in range(fit_input['N'])
        ]
        return sum(log_likelihoods)

    def grid_log_band_likelihood(self, fit_input, n_array, gamma_array):
        assert n_array.ndim == 1
        assert gamma_array.ndim == 1
        grid = np.empty((n_array.size, gamma_array.size))
        for i, n_s in enumerate(n_array):
            for j, gamma in enumerate(gamma_array):
                grid[i,j] = self.log_band_likelihood(fit_input, n_s, gamma)
        return grid

    def argmax_band_likelihood(self, fit_input, n_array, gamma_array):
        """ Return likelihood-maximizing (n_s,gamma) combination, i.e. ^n_s, ^γ
        """
        grid = self.grid_log_band_likelihood(fit_input, n_array, gamma_array)
        n_hat, g_hat = np.unravel_index(grid.argmax(), grid.shape)
        return n_hat, g_hat

def sqeuclidean(x):
    return np.inner(x, x).item()

