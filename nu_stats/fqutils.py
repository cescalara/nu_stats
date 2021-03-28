from collections import OrderedDict
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from iminuit import Minuit

from nu_stats.simulation import Simulation


class MarginalisedEnergyLikelihood:
    def __init__(
        self,
        is_source:bool = True,
        z:float = 0.0,
        Emin:float = 1e3,
        Emax:float = 1e10,
        n_Esim:int = 10000,
        min_index:float = 1.5,
        max_index:float = 4.0,
        Nbins:int = 50,
        prefab_file:str = None
    ):
        """
        Compute the marginalised energy likelihood by using a 
        simulation of a large number of reconstructed muon 
        neutrino tracks.
        
        :param energy: Reconstructed muon energies (preferably many).
        :param sim_index: Spectral index of source spectrum in sim.
        """
        self.is_source = is_source
        self.z = z

        self._min_index = min_index
        self._max_index = max_index

        self.Emin = Emin
        self.Emax = Emax
        self.n_Esim = n_Esim

        self._min_E = Emin/100 # Because Edet min can be lower than PL min
        self._max_E = Emax*10  # Because Edet max can be higher than PL max

        self._index_bins = np.linspace(min_index, max_index, Nbins)
        self._ibin_width = self._index_bins[1] - self._index_bins[0]
        self._energy_bins = np.linspace(np.log10(self._min_E),
                                        np.log10(self._max_E),
                                        Nbins)
        self._Ebin_width = self._energy_bins[1] - self._energy_bins[0]
        self._Nbins = Nbins

        if prefab_file is None:
            self._precompute_histograms()
        else:
            ''' 
            It must be ensured that the other inputs are the same as those
            that generated the likelihood matrix.
            TODO: make this easier, through e.g. hdf or pkl
            '''
            with open(prefab_file, 'rb') as f:
                self._likelihood = np.load(f)
                assert self._likelihood.shape == (Nbins-1, Nbins-1)
        
    def save_histogram(self, filename:str = None):
        if filename is None:
            filename = f'tmp/{self.is_source}{self.z}_{self._Nbins}lik.npy'
        with open(filename, 'wb') as f:
            np.save(f, self._likelihood)

    def _precompute_histograms(self):
        self._likelihood = np.zeros(
            (len(self._index_bins[:-1]), len(self._energy_bins[:-1]))
        )
        for i, index in enumerate(self._index_bins[:-1]):
            index_center_bin = index + (self._index_bins[i + 1] - index) / 2
            print('\r', f'Running sim {i+1}/{self._Nbins-1}', end='')
            Esim = self.Esim = Simulation(
                self.is_source*u.erg/u.s,
                index_center_bin, 
                self.z, 
                (1-self.is_source)/(u.GeV * u.cm**2 * u.s), 
                self.Emin*u.GeV, 
                self.Emax*u.GeV, 
                self.Emin*u.GeV, # assune Enorm = Emin for now
                self.n_Esim)
            Esim.run(seed = i) # Not ideal seeding
            energy = self.Esim.get_data()['Edet']

            hist, _ = np.histogram(
                np.log10(energy),
                bins=self._energy_bins,
                density=True, 
            )
            # because density=True, hist gives the value of the pdf at the bin,
            # normalized such that the integral over the range is 1
            self._likelihood[i] = hist
        print('\r')

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
        
        if E == self._max_E:
            E_index = self._Nbins-2
        else:
            E_index = np.digitize(np.log10(E), self._energy_bins)-1

        if new_index == self._max_index:
            i_index = self._Nbins-2
        else:
            i_index = np.digitize(new_index, self._index_bins)-1

        if interpol == 1 or interpol == 2:
            '''
            Perform linear interpolation between closest bins.
            This is needed for minos to work (mapping directly to historgam has
            only dirac-like gradients)
            '''
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
                lik = ((1-dE)*f0 + (dE)*f1)/ self._Nbins
                return max(lik, 1e-10)

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

            return max(lik, 1e-10)

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
    def __init__(self):
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
        self.spacial_only = True # Spacial only until energy likelihoods are given
    
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
        regard for spacial factor, we can use a single source with any L.
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
        self.spacial_only = False
        print('Marginalized energy likelihood generated.')

    def construct_bg_energy_likelihood(self,
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
        self.spacial_only = False
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

        spacial_factor = (kappa/np.pi
            * np.exp(-kappa
                    * sqeuclidean(source_dir - obs_dir)/2
                    )
                ) # bivariate normal with kappa = 1/sigma^2
        if spacial_factor == 0: raise ValueError
        if self.spacial_only:
            return spacial_factor
        else:
            assert hasattr(self, 'energy_likelihood'), 'Missing P(E_d|γ, src)'
            energy_factor = self.energy_likelihood(E_d, gamma) # p(E_d|gamma)
            if energy_factor == 0: raise ValueError(
                f'P(E_d|Υ,src) gave 0.0 for E_d={E_d}, Υ={gamma}')
            return spacial_factor * energy_factor

    def bg_likelihood(self, E_d, gamma):
        """ Evaluate Bg likelihood
        Args:
            source_dir (np.ndarray): direction of source
            obs_dir (np.ndarray): direction of observed neutrino
            kappa (float): uncertainty of neutrino direction in obs
        """
        spacial_factor = 1/(4*np.pi) # uniform on sphere
        if self.spacial_only:
            return spacial_factor
        else:
            assert hasattr(self, 'bg_energy_likelihood'), 'Missing P(E_d|γ, bg)'
            return spacial_factor * self.bg_energy_likelihood(E_d, gamma)
    

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
            # print(f)
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
            if self.spacial_only:
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
            
            if self.spacial_only:
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