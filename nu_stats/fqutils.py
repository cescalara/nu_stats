from collections import OrderedDict
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from iminuit import Minuit

from nu_stats.energylikelihoods import AtmosphericEnergyLikelihood, MarginalisedEnergyLikelihood


class FqStructure:
    def __init__(self):
        """
        Class for holding useful stuff to do with the Braun paper approach.
        """
        self.spatial_only = True # spatial only until energy likelihoods are given
    
    def construct_source_energy_likelihood(self,
        z: float,
        Emin: u.GeV,
        Emax: u.GeV,
        min_index: float,
        max_index: float,
        n_Esim: int,
        Nbins: int,
        prefab_likelihood_file: str = None
    ):
        print('Generating marginalized energy likelihood..')
        '''
        Because we only want a sample of Edet for a certain gamma, with no
        regard for spatial factor, we can use a single source with any L.
        '''
        self.energy_likelihood = MarginalisedEnergyLikelihood(
            is_source = True,
            z = z,
            Emin = Emin.value,
            Emax = Emax.value,
            n_Esim = n_Esim,
            min_index = min_index,
            max_index = max_index,
            Nbins = Nbins,
            prefab_file = prefab_likelihood_file
        )
        self.spatial_only = False
        print('Marginalized energy likelihood generated.')

    def construct_diffuse_bg_energy_likelihood(self,
        z: float,
        Emin: u.GeV,
        Emax: u.GeV,
        min_index: float,
        max_index: float,
        n_Esim: int,
        Nbins: int,
        prefab_likelihood_file: str = None
    ):
        print('Generating marginalized background energy likelihood..')
        '''
        TODO: Because is_source=False here will make the reference
        energy simulations be set to bg only and z in the Simulation input is
        the source's redshift and the bg is hardcoded, z here won't actually do
        anything, but can be good to keep in case of restructuring.
        This comment is a warning that passing another z to the bg_EL will not
        necessarily do anything.
        As of 27th of March 1.0 is hardcoded as z_bg in Simulation so that is
        the redshift that will go into the bg_MEL.
        '''
        self.bg_energy_likelihood = MarginalisedEnergyLikelihood(
            is_source = False, 
            z = z,
            Emin = Emin.value,
            Emax = Emax.value,
            n_Esim = n_Esim,
            min_index = min_index,
            max_index = max_index,
            Nbins = Nbins,
            prefab_file = prefab_likelihood_file
        )
        self.spatial_only = False
        print('Separate marginalized energy likelihood generated for bg.')
    
    def construct_atm_bg_energy_likelihood(self,
        Emin: u.GeV,
        Emax: u.GeV,
        n_Esim: int,
        Nbins: int,
        prefab_likelihood_file: str = None
    ):
        print('Generating atmospheric background energy likelihood..')
        '''
        Parameters like index and redshift are fixed in the Simulation class
        '''
        self.bg_energy_likelihood = AtmosphericEnergyLikelihood(
            Emin = Emin.value,
            Emax = Emax.value,
            n_Esim = n_Esim,
            Nbins = Nbins,
            prefab_file = prefab_likelihood_file
        )
        self.spatial_only = False
        print('Separate marginalized energy likelihood generated for bg.')

    def set_fit_input(self, fit_input):
        self.fit_input = fit_input

    def source_likelihood(self,
        E_d : float, # Event Edet
        obs_dir: np.ndarray, # Event
        gamma: float, # Source
        source_dir: np.ndarray, # Source
        kappa: float # Detector
    ):
        """ Evaluate source likelihood
        Args:
            E_d
            obs_dir (np.ndarray): direction of observed neutrino
            gamma (float): spectral index
            source_dir (np.ndarray): direction of source
            kappa (float): uncertainty of neutrino direction in obs
        """
        assert source_dir.ndim == 1, \
            f'source_dir.ndim = {source_dir.ndim}, expected 1'
        assert obs_dir.ndim == 1, \
            f'obs_dir.ndim = {obs_dir.ndim}, expected 1'

        spatial_factor = (kappa/np.pi
            * np.exp(-kappa
                    * sqeuclidean(source_dir - obs_dir)/2
                    )
                ) # bivariate normal with kappa = 1/sigma^2
        if spatial_factor == 0: raise ValueError
        if self.spatial_only:
            return spatial_factor
        else:
            assert hasattr(self, 'energy_likelihood'), 'Missing P(E_d|γ, src)'
            energy_factor = self.energy_likelihood(E_d, gamma) # p(E_d|gamma)
            if energy_factor == 0: raise ValueError(
                f'P(E_d|Υ,src) gave 0.0 for E_d={E_d}, Υ={gamma}')
            return spatial_factor * energy_factor

    def bg_likelihood(self, E_d, gamma):
        """ Evaluate Bg likelihood
        Args:
            source_dir (np.ndarray): direction of source
            obs_dir (np.ndarray): direction of observed neutrino
            kappa (float): uncertainty of neutrino direction in obs
        """
        spatial_factor = 1/(4*np.pi) # uniform on sphere
        if self.spatial_only:
            return spatial_factor
        elif hasattr(self, 'bg_energy_likelihood'):
            return spatial_factor * self.bg_energy_likelihood(E_d, gamma)    

    if True: # Foldable section for methods on individual events
        def test_stat(self,
            E_d : np.ndarray, # Edets from sim
            obs_dir: np.ndarray, # det_dir from sim
            gamma: float,
            source_dir: np.ndarray,
            kappa: float # Detector
        ):
            N_obs = obs_dir.shape[0]
            TS = np.empty(N_obs)
            TS[:] = np.NaN
            for j in range(N_obs):
                S = self.source_likelihood(
                    E_d[j],
                    obs_dir[j],
                    gamma,
                    source_dir,
                    kappa
                    )
                LR = S / self.bg_likelihood(E_d[j], gamma)
                TS[j] = 2*np.log(LR)
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
            self.fit_input = fit_input

            sim_TS = self.test_stat(
                self.fit_input['Edet'],
                self.fit_input['det_dir'],
                gamma,
                self.fit_input['source_dir'],
                self.fit_input['kappa']
                )
            bg_TS = self.test_stat(
                bg_dat['Edet'],
                bg_dat['det_dir'],
                gamma,
                self.fit_input['source_dir'],
                bg_dat['kappa']
                )

            ## get p values
            sim_p = np.zeros_like(sim_TS)
            for i,obs_TS in enumerate(sim_TS):
                sim_p[i] = (np.mean(bg_TS >= obs_TS))
            return sim_TS, bg_TS, sim_p

        def event_source_likelihood_from_index(self,
            gamma_array: np.ndarray,
            event_idx: int,
            fit_input: OrderedDict = None
        ):
            if fit_input is not None:
                self.fit_input = fit_input
            elif not hasattr(self, 'fit_input'):
                raise AttributeError(
                'fit_input not set, pass it as fourth arg, or\n\
                enter it with set_fit_input, or set directly through self.fit_input'
                )

            assert event_idx < self.fit_input['N']
            assert gamma_array.ndim == 1
            self.sl_arr = np.empty(gamma_array.size)
            for j, gamma in enumerate(gamma_array):
                self.sl_arr[j] = self.source_likelihood(
                                        self.fit_input['Edet'][event_idx],
                                        self.fit_input['det_dir'][event_idx],
                                        gamma,
                                        self.fit_input['source_dir'],
                                        self.fit_input['kappa']
                                    )
            return self.sl_arr
    
    if True: # Foldable section for methods on full band
        def log_band_likelihood(self, n_s: int, gamma: float):
            '''
            Returns log(L(x_s,n_s,gamma)) (7) to be maximized w.r.t. n_s and gamma
            for thothereir estimates.
            i.e. sum_i(log(n_s/N S_i + (1-n_s/N) B_i)
            '''
            if not hasattr(self, 'fit_input'):
                raise AttributeError(
            'fit_input not set, enter it with set_fit_input or directly through self.fit_input'
            )

            f = n_s/self.fit_input['N']
            likelihoods = [f * self.source_likelihood(
                        self.fit_input['Edet'][i],
                        self.fit_input['det_dir'][i],
                        gamma,
                        self.fit_input['source_dir'],
                        self.fit_input['kappa']
                        )
                    + (1-f) * self.bg_likelihood(self.fit_input['Edet'][i], gamma)
                 for i in range(self.fit_input['N'])]
            
            log_likelihoods = np.log(likelihoods)
            return sum(log_likelihoods)#, log_likelihoods

        def _neg_lbl(self, n_s, gamma):
            return -self.log_band_likelihood(n_s, gamma)

        def minimize_neg_lbl(self):
            """
            Minimize -log(likelihood_ratio) for the source hypothesis, 
            returning the Minuit object, best fit ns and index.
            """
            if self.spatial_only:
                init_index = 0 # anything should work here
            else:
                init_index = np.mean((self.energy_likelihood._min_index,
                                      self.energy_likelihood._max_index))
            
            init_ns = int(np.arange(self.fit_input['N']).mean())

            m = Minuit(self._neg_lbl,
                n_s = init_ns,
                gamma = init_index)

            m.errors = [1, 0.1]
            m.errordef = Minuit.LIKELIHOOD # 0.5
            
            if self.spatial_only:
                m.fixed['gamma'] = True
                m.limits = [(0, self.fit_input['N']-1),
                            (None,None)]
            else:
                m.limits = [(0, self.fit_input['N']-1),
                            (self.energy_likelihood._min_index,
                             self.energy_likelihood._max_index)]

            m.migrad()

            self._best_fit_ns = m.values["n_s"]
            self._best_fit_index = m.values["gamma"]
            return m, self._best_fit_ns, self._best_fit_index

        def band_TS(self, n_hat, gamma_hat, gamma_0):
            """Get test statistic for best fitting n_s and index

            Args:
                n_hat: best fitting n_s
                gamma_hat: best fitting spectral index
                gamma_0: best fitting spectral index for n_s fixed at 0

            Returns:
                float: value of test statistic λ.
            """            
            TS = -2 * (self.log_band_likelihood(0, gamma_0)
                     - self.log_band_likelihood(n_hat, gamma_hat))
            # first log_band term is null hypothesis and 
            return TS


        def grid_log_band_likelihood(self,
            n_array: np.ndarray,
            gamma_array: np.ndarray,
            fit_input: OrderedDict = None
        ):
            if fit_input is not None:
                self.fit_input = fit_input
            elif not hasattr(self, 'fit_input'):
                raise AttributeError(
                    'fit_input not set, pass it as fourth arg, or\n\
                    enter it with set_fit_input, or set directly through self.fit_input'
                    )
            
            assert n_array.max() <= self.fit_input['N']
            assert n_array.ndim == 1
            assert gamma_array.ndim == 1
            self.lbl_grid = np.empty((n_array.size, gamma_array.size))
            for i, n_s in enumerate(n_array):
                for j, gamma in enumerate(gamma_array):
                    self.lbl_grid[i,j] = self.log_band_likelihood(n_s, gamma)
            self._n_ss = n_array
            self._gammas = gamma_array
            return self.lbl_grid

        def argmax_band_likelihood(self,
            n_array: np.ndarray = None,
            gamma_array: np.ndarray = None,
            fit_input: OrderedDict = None
        ):
            """ Return likelihood-maximizing (n_s,gamma) combination, i.e. ^n_s, ^γ
            """
            if n_array is None or gamma_array is None:
                assert (n_array is None) == (gamma_array is None),\
                    'n_array and gamma_array must othereither both or none be passed'
                assert fit_input is None,\
                    'leave None if same fit input, or pass arrays for n and gamma'
                assert hasattr(self, 'lbl_grid'),\
                    'A grid of log band likelihoods is needed\n\
                    pass arrays of n and gamma'
            elif fit_input is not None:
                self.fit_input = fit_input
            else:
                if not hasattr(self, 'fit_input'):
                    raise AttributeError(
                        'fit_input not set, pass it as fourth arg, or\n\
                        enter it with set_fit_input, or set directly through self.fit_input'
                        )
                self._n_ss = n_array
                self._gammas = gamma_array
                self.grid_log_band_likelihood(n_array, gamma_array)

            n_hat_idx, g_hat_idx = np.unravel_index(self.lbl_grid.argmax(),
                                            self.lbl_grid.shape)
            self._grid_best_fit_ns = self._n_ss[n_hat_idx]
            self._grid_best_fit_index = self._gammas[g_hat_idx]
            return self._grid_best_fit_ns, self._grid_best_fit_index
        
        def plot_lbl_contour(self, show = False):
            if not hasattr(self, 'lbl_grid'):
                raise AttributeError(
                    'Missing previous grid_log_band_likelihood call'
                    )
            plt.contour(self._gammas,self._n_ss, self.lbl_grid, 10)
            plt.xlabel('γ')
            plt.ylabel('n_s')
            plt.colorbar()
            n_h, g_h = self.argmax_band_likelihood()
            plt.scatter(g_h, n_h, marker='x', c ='r')
            plt.title('Band log-likelihhod vs spectral index and n_s')
            if show: plt.show()
        


## Helper Functions -----------------------------------------------------------
def sqeuclidean(x):
    return np.inner(x, x).item()


def plot_loghist(x, bins, *args, **kwargs):
    logbins = np.logspace(np.log10(min(x)),np.log10(max(x)), bins+1)
    h = plt.hist(x, bins=logbins, *args, **kwargs)[0:2]
    plt.xscale('log')
    return h