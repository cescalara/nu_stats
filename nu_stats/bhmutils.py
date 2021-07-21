import numpy as np
import arviz, corner
from cmdstanpy import CmdStanModel

class BhStructure:
    def __init__(self, fit_input:dict=None):
        """
        Class for holding useful stuff to do with the Braun atricle approach to
        neutrino source association.
        Wraps a couple of stan function calls.
        """
        self.fit_input = fit_input
        

    def set_fit_input(self, fit_input:dict):
        """
        Args:
            fit_input (dict)
        """
        self.fit_input = fit_input

    def load_model(self, stan_file_name:str):
        self.stan_model = CmdStanModel(stan_file = stan_file_name)
    
    def run(self, num_chains:int = 4, samples_per_chain:int = 1000, **kwargs):
        """ Run the NUTS sampler to sample from model posterior through
            cmdstanpy.CmdStanModel.sample on the loaded stan model.

        Args:
            num_chains (int, optional): number of chains for sampler.
                Defaults to 4.
            samples_per_chain (int, optional): number of samples to generate per chain.
                Defaults to 1000.
            Additional optional args are also passed 
        """        
        self.fit = self.stan_model.sample(
            data=self.fit_input,
            iter_sampling=samples_per_chain,
            chains=num_chains,
            **kwargs
        )
        self.vars = self.fit.stan_variables()
    
    def print_diagnostic(self):
        self.fit.diagnose()

    def print_summary(self):
        print(self.fit.summary())
    
    def plot_traces(self, var_names:list, **kwargs):
        arviz.plot_trace(self.fit, var_names=var_names, **kwargs)

    def plot_corner(self, var_names:list, truths_list:list, **kwargs): 
        samples = np.column_stack([self.vars[key] for key in var_names])
        corner.corner(samples, labels=var_names, truths=truths_list, **kwargs)

    def association_probs(self, expected:bool=True):
        """ Get association probabilities for fitted events
        Args:
            expected (bool, optional):
                True: Use the expected values from either chains then normalize
                False: Normalize element-wise
                Note that these two are very different
        Returns:
            np.array: association probabilities (estimate)
        """
        N_events = self.fit_input['N']
        log_probs = self.fit.stan_variable('log_prob')
        log_probs = log_probs.transpose(1,2,0)
        # logprobs shape: (N_observations, N_components, N_samples)

        n_comps = np.shape(log_probs)[1] # number of components

        if expected:
            association_prob = np.zeros((N_events,n_comps))
            association_prob.fill(np.nan)
            for i, lp in enumerate(log_probs): # for each observation...
                ups = [] # unnormalized association prob
                ps = [] # association prob
                for c in range(n_comps): # for each component
                    ups.append(np.sum(np.exp(lp[c])))
                norm = sum(ups)

                for c in range(n_comps):
                    ps.append(ups[c] / norm)
                association_prob[i,:] = np.array(ps)

            # Unnecessary sanity check
            assert not np.isnan(association_prob).any()
            return association_prob

        else:
            normalized_probs = np.zeros_like(log_probs)
            for j, lp in enumerate(log_probs):
                nlp = []
                for c in range(n_comps):
                    nlp.append(np.exp(lp[c]))
                norm = sum(nlp)
                normalized_probs[j] = np.exp(lp)/norm # elementwise divide
            return normalized_probs