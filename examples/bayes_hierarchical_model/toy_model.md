---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.0
  kernelspec:
    display_name: nu_stats
    language: python
    name: nu_stats
---

## Toy BHM

```python
from astropy import units as u
import numpy as np
from cmdstanpy import CmdStanModel
import arviz, corner

import sys
sys.path.append("../../")
from nu_stats.simulation import Simulation
```

## Simulation

Simulate some fake data using the `nu_stats.simulation` module. The simulation could also be run in stan, as shown in the `stan/toy_hbm_sim.stan` code. 

```python
# Choose simulation parameters
L = 1e48 * (u.erg/u.s)
gamma = 2.2
z = 0.3
F_diff_norm = 1e-16 * 1/(u.GeV * u.cm**2 * u.s)
Emin = 1e5 * u.GeV
Emax = 1e8 * u.GeV
Enorm = 1e5 * u.GeV
```

```python
sim = Simulation(L, gamma, z, F_diff_norm, Emin, Emax, Enorm)
sim.run(seed=42)
```

```python
sim.show_spectrum()
```

```python
sim.show_skymap()
```

```python
# Extract simulated data and get info needed for fit
fit_input = sim.get_fit_input()
```

## Fit
Fit the simulated data using the Stan model `toy_bhm.stan`.

```python
stan_model = CmdStanModel(stan_file="stan/toy_bhm.stan")
fit = stan_model.sample(data=fit_input, iter_sampling=1000, chains=4, seed=42)
```

```python
# Trace plot
var_names = ["L", "F_diff", "f", "gamma"]
arviz.plot_trace(fit, var_names=var_names);
```

```python
# Corner plot, comparing with truth form sim
chain = fit.stan_variables()
samples = np.column_stack([chain[key].values.T[0] for key in var_names])
truths_list = [sim.truth[key] for key in var_names]

corner.corner(samples, labels=var_names, truths=truths_list);
```

```python

```
