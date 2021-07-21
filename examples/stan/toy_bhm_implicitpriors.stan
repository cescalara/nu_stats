functions {

#include vMF.stan
#include utils.stan
  
}

data {

  /* Neutrino data */
  int N;
  vector[N] Edet; // GeV
  unit_vector[3] det_dir[N];

  /* Source info */
  unit_vector[3] source_dir;
  real D; // m
  real z;
  real z_bg;
  real Emin; // GeV
  real Emax; // GeV
  
  /* Detector info */
  real T; // s
  real kappa;
  real aeff; // m^2

}

parameters {

  real<lower=1, upper=3> gamma;
  
  real<lower=0, upper=1e55> L; // GeV s^-1
  
  real<lower=0, upper=1e-5> F_bg; // m^-2 s^-1
  
  vector<lower=Emin, upper=Emax>[N] Etrue;
  
}

transformed parameters {

  real<lower=0, upper=1> f;
  real F_src;
  vector[2] F;
  real Nex;
  real Nex_ps;
  real Nex_bg;
  vector[2] log_prob[N];
  vector[N] Earr;
  
  F_src = L / (4 * pi() * pow(D, 2)); // GeV m^-2 s^-1
  F_src *= flux_conv(gamma, Emin, Emax); // m^-2 s^-1

  F[1] = F_src;
  F[2] = F_bg;

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

  /* Calculate expected number of events */
  Nex_ps = T * aeff * F_src; 
  Nex_bg = T * aeff * F_bg; 
  Nex = Nex_ps + Nex_bg;

}

model {

  /* Add rate contribution to Stan's target density */
  for (i in 1:N) {

    target += log_sum_exp(log_prob[i]);

  }

  /* Poisson process normalisation */
  target += -Nex;

  /* Uniform priors (default when not explicit)*/
  
}
