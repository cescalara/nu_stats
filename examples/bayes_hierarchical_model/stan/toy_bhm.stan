functions {

#include vMF.stan
#include interpolation.stan
  
  /**
   * To convert from units of GeV m^-2 s^-1 to 
   * m^-2 s^-1.
   */
  real flux_conv(real gamma, real e_low,real e_up) {

    real f1;
    real f2;

    if(gamma == 1.0) {
      f1 = (log(e_up)-log(e_low));
    }
    else {
      f1 = ((1/(1-gamma))*((e_up^(1-gamma))-(e_low^(1-gamma))));
    }
    
    if(gamma == 2.0) {
      f2 = (log(e_up)-log(e_low));
    }
    else {
      f2 = ((1/(2-gamma))*((e_up^(2-gamma))-(e_low^(2-gamma))));
    }
    
    return (f1/f2);
  }

  /**
   * Get exposure factor from interpolation
   * Units of [m^2 s]
   */
  vector get_exposure_factor(real gamma, vector gamma_grid, vector[] integral_grid, real T) {
    
    int K = num_elements(integral_grid);
    vector[K] eps;
    
    for (k in 1:K) {

      eps[k] = interpolate(gamma_grid, integral_grid[k], gamma) * T;
      
    }

    return eps;
  }

  /**
   * Calculate the expected number of detected events from each source.
   */
  real get_Nex(vector F, vector eps) {
    
    int K = num_elements(eps);
    real Nex = 0;
  
    for (k in 1:K) {
      Nex += F[k] * eps[k];
    }
  
    return Nex;
  }
  
}

/**
 * Calculate the expected number of detected events from each source.
 */
real get_Nex(vector F, vector eps) {
  
  int K = num_elements(eps);
  real Nex = 0;
  
  for (k in 1:K) {
    Nex += F[k] * eps[k];
  }
  
  return Nex;
}

data {

  /* Neutrino data */
  int N;
  vector[N] Edet; // in units of GeV
  unit_vector[3] det_dir[N];

  /* Source info */
  unit_vector[3] source_dir;
  real D; // in units of m
  real z;
  real z_bg;
  real Emin; // in units of GeV
  real Emax; //  in units of GeV
  
  /* Detector info */
  real T; // in units of s
  real kappa;

  /* For interpolation */
  int N_grid;
  vector[N_grid] gamma_grid;
  vector[N_grid] integral_grid[2];
  
}

parameters {

  real<lower=1, upper=4> gamma;
  
  real<lower=0, upper=1e52> L; // units of GeV s^-1
  
  real<lower=0, upper=1e-4> F_diff; // units of m^-2 s^-1
  
  vector<lower=Emin, upper=Emax>[N] Etrue;
  
}

transformed parameters {

  real<lower=0, upper=1> f;
  real F_src;
  vector[2] F;
  real Nex;
  vector[2] eps;
  vector[2] log_prob[N];
  vector[N] Earr;
  
  F_src = L / (4 * pi() * pow(D, 2)); // units of GeV m^-2 s^-1
  F_src *= fluc_conv(gamma, Emin, emax); // units of m^-2 s^-1

  F[1] = F_src;
  F[2] = F_diff;

  f = F_src / sum(F);

  /* Likelihood */
  for (i in 1:N) {

    /* Add flux weights */
    log_prob[i] = log(F);

    /* Same spectrum for both components */
    log_prob[i][k] += spectrum_lpdf(Etrue[i] | gamma, Emin, Emax);
      
    /* 1 <=> PS, 2 <=> BG */
    for (k in 1:2) {

      /* Point source */
      if (k == 1) {

	/* Energy losses */
	Earr[i] = Etrue[i] / (1 + z);

	/* P(det_dir | true_dir, kappa) */
	log_prob[i][k] += vMF_lpdf(det_dir[i] | source_dir, kappa);
	
      }

      /* Diffuse background */
      else if (k == 2) {

	/* Energy losses */
	Earr[i] = Etrue[i] / (1 + z_bg);

	/* P(uniform on sphere) */
	log_prob[i][k] += log(1 / (4*pi()));	
       
      }

    }

    /* Detection effects */
    log_prob[i][k] += lognormal_lpdf(Edet[i] | log(Earr[i]), 0.5);

  }

  eps = get_exposure_factor(gamma, gamma_grid, integral_grid, T);
  Nex = get_Nex(F, eps);
  
}

model {

  /* Add rate contribution to Stan's target density */
  for (i in 1:N) {

    target += log_sum_exp(log_prob[i])

  }

  /* Poisson process normalisation */
  target += -Nex;

  /* Weakly informative priors */
  L ~ lognormal(log(1e47), 5);
  F_diff ~ lognormal(log(1e-7), 5);
  gamma ~ normal(2, 2);
  
}
