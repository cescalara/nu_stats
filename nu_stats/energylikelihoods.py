import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from nu_stats.simulation import Simulation


class AtmosphericEnergyLikelihood:
    def __init__(
        self,
        Emin:float = 1e3,
        Emax:float = 1e10,
        n_Esim:int = 10000,
        Nbins:int = 50,
        prefab_file:str = None
    ):
        """ Compute and store the Energy likelihood for atmospheric background
        by simulating of a large number of reconstructed neutrino tracks.

        Args:
        Passed to Simulation: 
            Emin (float, optional): Doubles as Enorm Defaults to 1e3.
            Emax (float, optional): Defaults to 1e10.
            n_Esim (int, optional): Number of simulated tracks Defaults to 10000.
        Histogram params:
            Nbins (int, optional): Defaults to 50.
            prefab_file (str, optional): Name of file containing a previously
                made array. Defaults to None.
        """
        self.Emin = Emin
        self.Emax = Emax
        self.n_Esim = n_Esim

        self._min_E = Emin/100 # Because Edet min can be lower than PL min
        self._max_E = Emax*10  # Because Edet max can be higher than PL max

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
                assert self._likelihood.shape == (Nbins-1,)
        
    def save_histogram(self, filename:str = None):
        if filename is None:
            filename = f'tmp/atmospheric_{self._Nbins}lik.npy'
        with open(filename, 'wb') as f:
            np.save(f, self._likelihood)

    def _precompute_histograms(self):
        print('Running Esim')
        Esim = self.Esim = Simulation(
            atm_flux_norm = 2.5e-18 /(u.GeV * u.cm**2 * u.s), 
            Emin = self.Emin*u.GeV, 
            Emax = self.Emax*u.GeV, 
            Enorm = self.Emin*u.GeV, # assune Enorm = Emin for now
            N_events = self.n_Esim)
        Esim.run(seed = np.random.randint(100))
        energy = self.Esim.get_data()['Edet']

        hist, _ = np.histogram(
            np.log10(energy),
            bins=self._energy_bins,
            density=True, 
        )
        # because density=True, hist gives the value of the pdf at the bin,
        # normalized such that the integral over the range is 1
        self._likelihood = hist
        assert self._likelihood.shape == (len(self._energy_bins[:-1]),)
        print('done')

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

    def __call__(self, E, index_dummy, interpol = 1):
        """
        P(Ereco | index) = \int dEtrue P(Ereco | Etrue) P(Etrue | index)
        """

        if E < self._min_E or E > self._max_E:
            raise ValueError(
                f"Energy {E} is not in the accepted range"
                + f" between {self._min_E} and {self._max_E}"
            )

        if E == self._max_E:
            E_index = self._Nbins-2
        else:
            E_index = np.digitize(np.log10(E), self._energy_bins)-1

        if interpol == 1:
            '''
            Perform linear interpolation between closest bins.
            This is needed for minos to work (mapping directly to historgam has
            only dirac-like gradients)
            '''
            f0 = self._likelihood[E_index]
            midE = self._energy_bins[E_index]+self._Ebin_width/2
            low_E = np.log10(E) < midE
            otherEi = self._other_index(E_index, low_E)

            if otherEi is None:
                f1 = f0
            else:
                f1 = self._likelihood[otherEi]

            dE = ((np.log10(E) - midE) / self._Ebin_width)
            dE -= 2*dE*low_E # switch sign if to the low

            # linear interpolation
            lik = ((1-dE)*f0 + (dE)*f1)/ self._Nbins
            return max(lik, 1e-10)
        else:
            return self._likelihood[E_index]

    def plot_pdf(self, interp=1, scaled_plot = True, **kwargs):
        Es = np.logspace(np.log10(self._min_E),
                 np.log10(self._max_E), 10000)
        ll = [self(E,0, interp) for E in Es]
        if interp and not scaled_plot:
            ll = [l*self._Nbins for l in ll]
        p = plt.plot(Es, ll, **kwargs)
        plt.xscale('log')
        return p


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
        """[summary]

        Args:
            is_source (bool, optional): Simulate a point source?
                Defaults to True. If False, simulate diffuse bg
            z (float, optional): redshift of source, does not affect diffuse bg
                ass that is hardcoded in Simulation. Defaults to 0.0.
            Emin (float, optional): Passed to sims and used for bin lims.
                Defaults to 1e3.
            Emax (float, optional):Passed to sims and used for bin lims.
                Defaults to 1e10.
            n_Esim (int, optional): For each spectral index (γ) bin, how many 
                simulated events. Defaults to 10000.
            min_index (float, optional): Lower bin lim for spectral index.
                Defaults to 1.5.
            max_index (float, optional): Upper bin lim for spectral index.
                Defaults to 4.0.
            Nbins (int, optional): number of bins in both E and γ.
                Defaults to 50.
            prefab_file (str, optional): Name of file containing a previously
            made array. Defaults to None.
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
                L = self.is_source*u.erg/u.s,
                gamma = index_center_bin, 
                z = self.z, 
                F_diff_norm = (1-self.is_source)/(u.GeV * u.cm**2 * u.s), 
                Emin = self.Emin*u.GeV, 
                Emax = self.Emax*u.GeV, 
                Enorm = self.Emin*u.GeV, # assune Enorm = Emin for now
                N_events = self.n_Esim)
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

    def plot_pdf_at_idx(self, index, interp=2, scaled_plot = True, **kwargs):
        Es = np.logspace(np.log10(self._min_E),
                 np.log10(self._max_E), 10000)
        ll = [self(E, index, interp) for E in Es]
        if interp and not scaled_plot:
            ll = [l*self._Nbins for l in ll]
        p = plt.plot(Es, ll, **kwargs)
        plt.xscale('log')
        return p

    def plot_pdf_at_E(self, E, interp=2, scaled_plot = True, **kwargs):
        idx = np.linspace(self._min_index,
                 self._max_index,10000)
        ll = [self(E, index, interp) for index in idx]
        if interp and not scaled_plot:
            ll = [l*self._Nbins for l in ll]
        p = plt.plot(idx, ll, **kwargs)
        return p

    def plot_pdf_meshgrid(self, interp=2, **kwargs):
        idx = np.linspace(self._min_index,
                 self._max_index,100)
        Es = np.logspace(np.log10(self._min_E),
                 np.log10(self._max_E),100)
        Es, idx = np.meshgrid(Es, idx)
        vfunc = np.vectorize(self.__call__)
        Z = vfunc(Es, idx, interp)
        # np.savetxt("pdfmesh.csv", Z, delimiter=",")
        plt.pcolormesh(Es, idx, Z, shading='auto', **kwargs)
        plt.xscale('log')
