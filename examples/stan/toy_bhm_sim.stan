functions{

#include vMF.stan
#include utils.stan

}

data {

  real L; // GeV s^-1
  unit_vector[3] source_dir;
  real gamma;
  real F_bg; // m^-2 s^-1
  real Emin; // GeV
  real Emax; // GeV
  real z;
  real z_bg;
  real D; // m^2
  
  real T; // s
  real aeff; // m^2
  real kappa;
  
}

transformed data {

  real F_src;
  real f;
  real Nex_ps;
  real Nex_bg;
  real Nex;
  vector[2] w;
  int N;

  F_src = L / (4 * pi() * pow(D, 2));
  F_src *= flux_conv(gamma, Emin, Emax);

  f = F_src / (F_src + F_bg);

  Nex_ps = T * aeff * F_src; 
  Nex_bg = T * aeff * F_bg;
  Nex = Nex_ps + Nex_bg;


  N = poisson_rng(Nex);

  w[1] = Nex_ps / Nex;
  w[2] = Nex_bg / Nex;

}

generated quantities {

  vector[N] Etrue;
  vector[N] Earr;
  vector[N] Edet;
  unit_vector[3] true_dir[N];
  unit_vector[3] det_dir[N];
  vector[N] label;
  
  for (i in 1:N) {

    label[i] = categorical_rng(w);

    Etrue[i] = spectrum_rng(gamma, Emin, Emax);

    if (label[i] == 1) {

      Earr[i] = Etrue[i] / (1+z);
      true_dir[i] = source_dir;
      
    }
    else if (label[i] == 2) {

      Earr[i] = Etrue[i] / (1+z_bg);
      true_dir[i] = vMF_rng(source_dir, 0);
      
    }

    Edet[i] = lognormal_rng(log(Earr[i]), 0.5);
    det_dir[i] = vMF_rng(true_dir[i], kappa);

  }

}
