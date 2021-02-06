functions {

#include vMF.stan
#include interpolation.stan
#include utils.stan
  
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
  real aeff; // in units of m^2
}

parameters {

  real<lower=1, upper=4> gamma;
  
  real<lower=0, upper=1e55> L; // units of GeV s^-1
  
  real<lower=0, upper=1e-5> F_diff; // units of m^-2 s^-1
  
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
  F_src *= flux_conv(gamma, Emin, Emax); // units of m^-2 s^-1

  F[1] = F_src;
  F[2] = F_diff;

  f = F_src / sum(F);

  /* Likelihood */
  for (i in 1:N) {

    /* Add flux weights */
    log_prob[i] = log(F);
  
    /* 1 <=> PS, 2 <=> BG */
    for (k in 1:2) {

      /* Same spectrum for both components */
      log_prob[i][k] += spectrum_lpdf(Etrue[i] | gamma, Emin, Emax);
      
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

      /* Detection effects */
      log_prob[i][k] += lognormal_lpdf(Edet[i] | log(Earr[i]), 0.5);

    }

  }

  eps = get_exposure_factor(gamma, T, aeff, z, z_bg);
  Nex = get_Nex(F, eps);
  
}

model {

  /* Add rate contribution to Stan's target density */
  for (i in 1:N) {

    target += log_sum_exp(log_prob[i]);

  }

  /* Poisson process normalisation */
  target += -Nex;

  /* Weakly informative priors */
  L ~ lognormal(log(1e51), 5);
  F_diff ~ lognormal(log(1e-6), 5);
  gamma ~ normal(2, 2);
  
}
