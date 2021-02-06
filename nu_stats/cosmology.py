from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def luminosity_distance(z):
    """
    Luminosity distance in Mpc for given z.
    """

    return cosmo.luminosity_distance(z)
