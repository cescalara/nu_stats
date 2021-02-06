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

Simulate some fake data using the `nu_stats.simulation` module.

```python
# Choose simulation parameters
L = 5e48 * (u.erg/u.s)
gamma = 2.2
z = 0.3
F_diff_norm = 1e-15 * 1/(u.GeV * u.cm**2 * u.s)
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
```

```python
fit = stan_model.sample(data=fit_input, iter_sampling=1000, chains=4, seed=42)
```

```python
fit
```

```python
# Trace plot
var_names = ["L", "F_diff", "gamma", "f"]
arviz.plot_trace(fit, var_names=var_names);
```

```python
# Corner plot
chain = fit.stan_variables()
samples = np.column_stack([chain[key].values.T[0] for key in var_names])
truths_list = [sim.truth[key] for key in var_names]

corner.corner(samples, labels=var_names, truths=truths_list);
```

```python
sim.truth["F_diff"]
```

```python
from matplotlib import pyplot as plt
```

```python
fig, ax = plt.subplots()
bins = 10**np.linspace(-8, -5)
ax.hist(np.random.lognormal(np.log(1e-6), 2, 1000), bins=bins)
ax.set_xscale("log")
```

```python
fig, ax = plt.subplots()
ax.hist(fit.stan_variable("Nex").values)
```

```python
np.mean(fit.stan_variable("F_src").values.T[0])
```

```python
sim.point_source.integrate(sim.Emin, sim.Emax).to(1 / (u.m**2 * u.s))
```

```python
fit_input
```

```python
np.mean(fit.stan_variable("F_diff").values.T[0])
```

```python
sim.diffuse_bg.integrate(sim.Emin, sim.Emax).to(1 / (u.s * u.m**2))
```

```python

```
