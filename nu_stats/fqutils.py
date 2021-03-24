from collections import OrderedDict
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from iminuit import Minuit

from nu_stats.simulation import Simulation


class MarginalisedEnergyLikelihood:
    def __init__(
        self,
        energy,
        sim_index = 1.5,
        min_index = 1.5,
        max_index = 4.0,
        min_E = 1e3,
        max_E = 1e10,
        Nbins = 50,
        z = 0
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

        self._index_bins = np.linspace(min_index, max_index, Nbins)
        self._ibin_width = self._index_bins[1] - self._index_bins[0]
        self._energy_bins = np.linspace(np.log10(min_E), np.log10(max_E), Nbins)  # GeV
        self._Ebin_width = self._energy_bins[1] - self._energy_bins[0]
        self._Nbins = Nbins

        self.z = z

        self._precompute_histograms()
        self.interpol2dfunc = None

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
            # because density=True, hist gives the value of the pdf at the bin,
            # normalized such that the integral over the range is 1
            self._likelihood[i] = hist

    def _calc_weights(self, new_index):
        return np.power(self._energy, (self._sim_index - new_index))

    def _other_index(self, bin_index:int, val_lt_bin_center:bool):
        """
        Args:
            bin_index (int): index of bin value belongs to
            val_lt_bin_center (bool): if the value is lower than that of the bin center

        Returns:
            None, bin_index-1 or bin_index+1 depending on index validity
        """
        cases = [None, bin_index-1, bin_index+1]
        return cases[(val_lt_bin_center * (bin_index > 0))
                    + 2 * ((not val_lt_bin_center) * (bin_index < self._Nbins-2))]

    def __call__(self, E, new_index, interpol = 2):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        if E < self._min_E or E > self._max_E:
            raise ValueError(
                f"Energy {E} is not in the accepted range"
                + f" between {self._min_E} and {self._max_E}"
            )

        if new_index < self._min_index or new_index > self._max_index:
            raise ValueError(
                "Spectral index {new_index} is not in the accepted range"
                + f" between {self._min_index} and {self._max_index}"
            )

        # rescale E to counteract the effect of redshift
        E = E*np.power((1+self.z),(self._sim_index - new_index))
        
        if E == self._max_E:
            E_index = self._Nbins-2
        else:
            E_index = np.digitize(np.log10(E), self._energy_bins)-1

        if new_index == self._max_index:
            i_index = self._Nbins-2
        else:
            i_index = np.digitize(new_index, self._index_bins)-1

        if interpol == 1 or interpol == 2:
            # perform linear interpolation between closest bins
            f0 = self._likelihood[i_index, E_index]
            midE = self._energy_bins[E_index]+self._Ebin_width/2
            low_E = np.log10(E) < midE
            otherEi = self._other_index(E_index, low_E)

            if otherEi is None:
                f1 = f0
            else:
                f1 = self._likelihood[i_index, otherEi]

            dE = ((np.log10(E) - midE) / self._Ebin_width)
            dE -= 2*dE*low_E # switch sign if to the low

            if interpol == 1:
                # linear interpolation
                return (1-dE)*f0 + (dE)*f1 / self._Nbins

            elif interpol == 2:
                # bilinear interpolation
                # f00, f01, f10, f11
                f = [f0, f0, f0, f0]
                midi = self._index_bins[i_index] + self._ibin_width/2
                low_i = new_index < midi
                otherii = self._other_index(i_index, low_i)

                if otherEi is not None:
                    f[1] = self._likelihood[i_index, otherEi]
                    f[3] = f[1] # will be kept if otherii is None

                if otherii is not None:
                    f[2] = self._likelihood[otherii, E_index]
                    f[3] = f[2] # will be kept if otherEi is None

                if (otherEi is not None) and (otherii is not None):
                    f[3] = self._likelihood[otherii, otherEi]

                di = ((new_index - midi) / self._ibin_width)
                di -= 2*di*low_i

                lik = (f[0] * (1-dE) * (1-di)
                    + f[1] * dE * (1-di)
                    + f[2] * (1-dE) * di
                    + f[3] * dE * di) / self._Nbins

            return lik

        else:
            return self._likelihood[i_index, E_index]

    def plot_pdf_at_idx(self, index, interp=2, scaled_plot = True):
        Es = np.logspace(np.log10(self._min_E),
                 np.log10(self._max_E), 10000)
        ll = [self(E, index, interp) for E in Es]
        if interp and not scaled_plot:
            ll = [l*self._Nbins for l in ll]
        p = plt.plot(Es, ll)
        plt.xscale('log')
        return p

    def plot_pdf_at_E(self, E, interp=2, scaled_plot = True):
        idx = np.linspace(self._min_index,
                 self._max_index,10000)
        ll = [self(E, index, interp) for index in idx]
        if interp and not scaled_plot:
            ll = [l*self._Nbins for l in ll]
        p = plt.plot(idx, ll)
        return p

    def plot_pdf_meshgrid(self, interp=2):
        idx = np.linspace(self._min_index,
                 self._max_index,100)
        Es = np.logspace(np.log10(self._min_E),
                 np.log10(self._max_E),100)
        Es, idx = np.meshgrid(Es, idx)
        vfunc = np.vectorize(self.__call__)
        Z = vfunc(Es, idx, interp)
        # np.savetxt("pdfmesh.csv", Z, delimiter=",")
        plt.pcolormesh(Es, idx, Z, shading='auto')
        plt.xscale('log')


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

        No Args: Spatial model only, i.e. P(Edet|gamma) = 1

        Args: Properties for simulation for precomputing P(Edet|gamma),
        passed to construct_source_energy_likelihood
            z (float, optional): Defaults to np.nan.
            Emin (u.GeV, optional): Defaults to np.nan.
            Emax (u.GeV, optional): Defaults to np.nan.
            Esim_gamma (float, optional):  Defaults to np.nan.
            n_Esim (int, optional): Defaults to np.nan.
            E_seed (int, optional): Defaults to 123.
        """    
        self.E_input_array = [z, Emin, Emax, Esim_gamma, n_Esim, E_seed]
        self.construct_source_energy_likelihood(*self.E_input_array)
    
    def construct_source_energy_likelihood(self,
        z: float = np.nan,
        Emin: u.GeV = np.nan,
        Emax: u.GeV = np.nan,
        Esim_gamma: float = np.nan,
        n_Esim: int = np.nan,
        E_seed = 123,
    ):
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
            self.energy_likelihood = lambda Edet, spectral_index: 1
            print('Marginalized energy likelihood taken to be 1.')
        else:
            print('Generating marginalized energy likelihood..')
            '''
            Because we only want a sample of Edet for a certain gamma, with no
            regard for spacial factor, we can use a single source with any L.
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
            self.Esim_dat = self.Esim.get_data()

            self.energy_likelihood = MarginalisedEnergyLikelihood(
                self.Esim_dat['Edet'],
                sim_index = Esim_gamma,
                min_index = Esim_gamma,
                max_index = 4.0,
                min_E = Emin.value/100, # Because Edet min can be lower than PL min
                max_E = Emax.value*10,
                Nbins = 50,
                z = z
            )
            print('Marginalized energy likelihood generated.')

    def construct_bg_energy_likelihood(self,
        z: float = np.nan,
        Emin: u.GeV = np.nan,
        Emax: u.GeV = np.nan,
        Esim_gamma: float = np.nan,
        n_Esim: int = np.nan,
        E_seed = 123
    ):
        print('Generating marginalized energy likelihood..')
        '''
        again we could use  we only want a sample of Edet for a certain gamma, with no
        regard for spacial factor, we can use a single source with any L.
        '''
        self.bgEsim = Simulation(
                0*u.erg/u.s,
                Esim_gamma, 
                z, 
                1/(u.GeV * u.cm**2 * u.s), 
                Emin, 
                Emax, 
                Emin, 
                n_Esim)
        self.bgEsim.run(seed = E_seed)
        self.bgEsim_dat = self.Esim.get_data()

        self.bg_energy_likelihood = MarginalisedEnergyLikelihood(
            self.bgEsim_dat['Edet'],
            sim_index = Esim_gamma,
            min_index = Esim_gamma,
            max_index = 4.0,
            min_E = Emin.value/100, # Because Edet min can be lower than PL min
            max_E = Emax.value*10,
            Nbins = 50,
            z = z
        )
        print('Separate marginalized energy likelihood generated for bg.')

    def set_fit_input(self, fit_input):
        self.fit_input = fit_input

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
        return spacial_factor * energy_factor

    def bg_likelihood(self, E_r, gamma):
        """
        If spacial is isotropic, normalisation doesn't matter
        since the TS is relative
        """
        spacial_factor = 1/(4*np.pi) # uniform on sphere
        if self.spacial_only:
            return spacial_factor
        elif hasattr(self, 'bg_energy_likelihood'):
            print('yes')
            return spacial_factor * self.bg_energy_likelihood(E_r, gamma)
        else:
            return spacial_factor * self.energy_likelihood(E_r, gamma)
    
    if True: # Foldable section for methods on individual events
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
                sim_p[i] = (np.mean(bg_TS > obs_TS))
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
            # print(f)
            log_likelihoods = [np.log(
                f * self.source_likelihood(
                    self.fit_input['Edet'][i],
                    self.fit_input['det_dir'][i],
                    gamma,
                    self.fit_input['source_dir'],
                    self.fit_input['kappa']
                    )
                + (1-f) * self.bg_likelihood(self.fit_input['Edet'][i], gamma)
                )
                for i in range(self.fit_input['N'])
            ]
            return sum(log_likelihoods)#, log_likelihoods

        def _neg_lbl(self, n_s, gamma):
            return -self.log_band_likelihood(n_s, gamma)

        def minimize_neg_lbl(self):
            """
            Minimize -log(likelihood_ratio) for the source hypothesis, 
            returning the best fit ns and index.
            Uses the iMiuint wrapper.
            """
            if isinstance(self.energy_likelihood, MarginalisedEnergyLikelihood):
                init_index = np.mean((self.energy_likelihood._min_index,
                                      self.energy_likelihood._max_index))
            else: # Spacial only
                init_index = 0 # anything should work here
            init_ns = int(np.arange(self.fit_input['N']).mean())

            m = Minuit(self._neg_lbl,
                n_s = init_ns,
                gamma = init_index)

            m.errors = [1, 0.1]
            m.errordef = Minuit.LIKELIHOOD # 0.5
            
            if isinstance(self.energy_likelihood, MarginalisedEnergyLikelihood):
                m.limits = [(0, self.fit_input['N']-1),
                            (self.energy_likelihood._min_index,
                             self.energy_likelihood._max_index)]
            else:
                m.fixed['gamma'] = True
                m.limits = [(0, self.fit_input['N']-1),
                            (None,None)]

            m.migrad()

            self._best_fit_ns = m.values["n_s"]
            self._best_fit_index = m.values["gamma"]
            return m, self._best_fit_ns, self._best_fit_index

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