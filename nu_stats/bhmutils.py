import numpy as np

def classify_events(fit_input, fit):
    """ Get association probabilities for fitted events
    Args:
        fit_input (Dict)
        fit (CmdStanPyMCMC)

    Returns:
        np.array: association probabilities
    """    
    N_events = fit_input['N']
    N_draws = fit.chains * fit.num_draws
    log_probs = fit.stan_variable('log_prob')
    log_probs = (log_probs.values
                 .reshape(N_draws, 2, N_events)
                 .transpose(2,1,0))

    n_comps = np.shape(log_probs)[1] # number of components
    
    association_prob = np.zeros((N_events,n_comps))
    association_prob.fill(np.nan)

    for i,lp in enumerate(log_probs):
        lps = []
        ps = []
        for src in range(n_comps):
            lps.append(np.mean(np.exp(lp[src]))) 
        norm = sum(lps)

        for src in range(n_comps):
            ps.append(lps[src] / norm)
        association_prob[i,:] = np.array(ps)

    # Unnecessary sanity check
    assert not np.isnan(association_prob).any()
    return association_prob