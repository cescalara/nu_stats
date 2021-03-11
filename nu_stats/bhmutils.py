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
    log_probs = fit.stan_variable('log_prob')
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