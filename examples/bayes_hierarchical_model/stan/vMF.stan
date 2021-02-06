/**
 * Stan functions for vMF sampling.
 *
 * @author Francesca Capel
 * @date February 2019
 */

/**
 * compute the absolute value of a vector 
 */
real abs_val(vector input_vector) {

  real av;
  int n = num_elements(input_vector);

  real sum_squares = 0;
  for (i in 1:n) {
    sum_squares += (input_vector[i] * input_vector[i]);
  }
  av = sqrt(sum_squares);
  return av;
    
}

/**
 * Sample point on sphere orthogonal to mu.
 */
vector sample_orthonormal_to_rng(vector mu) {

  int dim = num_elements(mu);
  vector[dim] v;
  vector[dim] proj_mu_v;
  vector[dim] orthto;
    
  for (i in 1:dim) {
    v[i] = normal_rng(0, 1);
  }
    
  proj_mu_v = mu * dot_product(mu, v) / abs_val(mu);
  orthto = v - proj_mu_v;
    
  return (orthto / abs_val(orthto));

}
  
/**
 * Rejection sampling scheme for sampling distance from center on
 * surface of the sphere.
 */
real sample_weight_rng(real kappa, int dim) {

  real sdim = dim - 1; /* as S^{n-1} */
  real b = sdim / (sqrt(4. * pow(kappa, 2) + pow(sdim, 2)) + 2 * kappa);
  real x = (1 - b) / (1 + b);
  real c = kappa * x + sdim * log(1 - pow(x, 2));
  
  int i = 0;
  real z;
  real w;
  real u;
  while (i == 0) {
    z = beta_rng(sdim / 2, sdim / 2);
    w = (1 - (1 + b) * z) / (1 - (1 - b) * z);
    u = uniform_rng(0, 1);
    if (kappa * w + sdim * log(1 - x * w) - c >= log(u)) {
      i = 1;
    }
  }

  return w;
}
  
/**
 * Generate an N-dimensional sample from the von Mises - Fisher
 * distribution around center mu in R^N with concentration kappa.
 */
vector vMF_rng(vector mu, real kappa) {

  int dim = num_elements(mu);
  vector[dim] result;
  
  real w = sample_weight_rng(kappa, dim);
  vector[dim] v = sample_orthonormal_to_rng(mu);

  result = ( v * sqrt(1 - pow(w, 2)) ) + (w * mu);
  return result;
   
}

/**
 * Sample a point uniformly from the surface of a sphere of 
 * a certain radius.
 */
vector sphere_rng(real radius) {
  
  vector[3] result;
  real u = uniform_rng(0, 1);
  real v = uniform_rng(0, 1);
  real phi = 2 * pi() * u;
  real theta = acos( (2 * v) - 1 );
  
  result[1] = radius * sin(theta) * cos(phi); 
  result[2] = radius * sin(theta) * sin(phi); 
  result[3] = radius * cos(theta);
  
  return result;
    
}

/*
 * Sample a point uniformly on the surface of a sphere of 
 * a certain radius. The v_lim option can be used to limit the
 * possible theta values.
 */
vector sphere_lim_rng(real radius, real v_lim) {

  vector[3] result;
  real u = uniform_rng(0, 1);
  real v = uniform_rng(v_lim, 1);

  real phi = 2 * pi() * u;
  real theta = acos(2 * v - 1);

  result[1] = radius * sin(theta) * cos(phi);
  result[2] = radius * sin(theta) * sin(phi);
  result[3] = radius * cos(theta);
  
  return result;
}

  
/**
 * Define the vMF PDF.
 * NB: Cannot be vectorised.
 * Uses sinh(kappa) ~ exp(kappa)/2 
 * approximation for kappa > 100.
 */
real vMF_lpdf(vector v, vector mu, real kappa) {
  
  real lprob;
  if (kappa > 100) {
    lprob = kappa * dot_product(v, mu) + log(kappa) - log(4 * pi()) - kappa + log(2);
  }
  else {
    lprob = kappa * dot_product(v, mu) + log(kappa) - log(4 * pi() * sinh(kappa));
  }
  return lprob;
    
}
