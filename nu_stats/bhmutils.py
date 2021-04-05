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
    
    def run(self, num_chains:int = 4, samples_per_chain:int = 1000, *args):
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
            *args
        )
        self.vars = self.fit.stan_variables()
    
    def print_diagnostic(self):
        self.fit.diagnose()

    def print_summary(self):
        self.fit.summary()
    
    def plot_traces(self, var_names:list):
        arviz.plot_trace(self.fit, var_names=var_names)

    def plot_corner(self, var_names:list, truths_list:list): 
        samples = np.column_stack([self.vars[key] for key in var_names])
        corner.corner(samples, labels=var_names, truths=truths_list)
  
    def classify_events(self):
        """ Get association probabilities for fitted events
        Returns:
            np.array: association probabilities
        """    
        N_events = self.fit_input['N']
        log_probs = self.fit.stan_variable('log_prob')
        log_probs = log_probs.transpose(1,2,0)
        # logprobs shape: (N_observations, N_components, N_samples)

        n_comps = np.shape(log_probs)[1] # number of components

        association_prob = np.zeros((N_events,n_comps))
        association_prob.fill(np.nan)

        for i, lp in enumerate(log_probs):
            lps = [] # unnormalized association prob
            ps = [] # association prob
            for src in range(n_comps):
                lps.append(np.sum(np.exp(lp[src]))) # Basically E[log_prob]
            norm = sum(lps)

            for src in range(n_comps):
                ps.append(lps[src] / norm)
            association_prob[i,:] = np.array(ps)

        # Unnecessary sanity check
        assert not np.isnan(association_prob).any()
        return association_prob