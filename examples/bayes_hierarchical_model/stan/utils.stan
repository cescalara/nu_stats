/**
 * Useful functions for Stan fits.
 *
 * @author Francesca Capel
 * @date February 2021
 */
  
/**
 * To convert from units of GeV m^-2 s^-1 to 
 * m^-2 s^-1.
 */
real flux_conv(real gamma, real e_low, real e_up) {

  real f1;
  real f2;
  
  if(gamma == 1.0) {
    
    f1 = log(e_up) - log(e_low);
    
  }
  
  else {
    
    f1 = (1.0 / (1.0-gamma)) * (pow(e_up, 1.0-gamma) - pow(e_low, 1.0-gamma));
    
  }
  
  if(gamma == 2.0) {
    
    f2 = log(e_up) - log(e_low);
    
  }
  
  else {
      
    f2 = (1.0 / (2.0-gamma)) * (pow(e_up, 2.0-gamma) - pow(e_low, 2.0-gamma));

  }
    
  return f1 / f2;
}

/**
 * Get exposure factor from interpolation
 * Units of [m^2 s]
 */
vector get_exposure_factor(real gamma, real T, real aeff, real z, real z_bg) {
  
  vector[2] eps;
  
  eps[1] = T * aeff * pow(1+z , -gamma);
  eps[2] = T * aeff * pow(1+z_bg, -gamma);
    
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

/**
 * Power law spectrum pdf.
 */
real spectrum_lpdf(real E, real gamma, real e_low, real e_up) {

  real N;
  real p;
  
  if(gamma == 1.0) {
    
    N = 1.0 / (log(e_up) - log(e_low));
    
  }
  else {
    
    N = (1.0 - gamma) / (pow(e_up, 1.0-gamma) - pow(e_low, 1.0-gamma));
    
  }
  
  p = N * pow(E, gamma*-1);
  
  return log(p);
    
}

/**
 * Generate random samples from a power law spectrum.
 */
real spectrum_rng(real gamma, real e_low, real e_up) {

  real uni_sample;
  real norm;
  
  norm = (1 - gamma) / (pow(e_up, 1-gamma) - pow(e_low, 1-gamma));
  uni_sample = uniform_rng(0, 1);
  
  return pow( (uni_sample*(1-gamma))/norm + pow(e_low, 1-gamma), 1/(1-gamma) );

}
