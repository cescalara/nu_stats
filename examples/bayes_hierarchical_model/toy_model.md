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

import sys
sys.path.append("../../")
from nu_stats.simulation import Simulation
```

## Simulation

Simulate some fake data using the `nu_stats.simulation` module.

```python
# Choose simulation parameters
L = 1e48 * (u.erg/u.s)
gamma = 2.2
z = 0.3
F_diff_norm = 1e-15 * 1/(u.GeV * u.cm**2 * u.s)
Emin = 1e5 * u.GeV
Emax = 1e8 * u.GeV
Enorm = 1e5 * u.GeV
```

```python
sim = Simulation(L, gamma, z, F_diff_norm, Emin, Emax, Enorm)
sim.run()
```

```python
sim.show_spectrum()
```

```python
sim.show_skymap()
```

## Fit
Fit the simulated data using the Stan model `toy_bhm.stan`.

```python

```
