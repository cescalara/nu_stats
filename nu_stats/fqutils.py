import numpy as np
from astropy.coordinates import SkyCoord
from .plotting import SphericalCircle
from matplotlib import pyplot as plt
from astropy import units as u


def omega_s(t: float):
    """ Describes the model-dependent relation between
        the γ-ray emission and the expected neutrino flux
        from source s as a function of time

    Args:
        t (float): time
    """
    return 1


def omega_acc(zenith_angle: float):
    """ Detector acceptance as a function of zenith angle
        (normalized over all zenith angles, θs)

    Args:
        zenith_angle (float)
    """
    return 1


def sqeuclidean(x):
    return np.inner(x, x).item()


def source_likelihood(source_dir: np.ndarray, obs_dir: np.ndarray, kappa: float):
    """ Evaluate source likelihood

    Args:
        source_dir (np.ndarray): direction of source
        obs_dir (np.ndarray): direction of observed neutrino
        kappa (float): uncertainty of neutrino direction in obs
    """
    assert source_dir.shape[0] == 1, \
        f'Source dir shape[0]={source_dir.shape[0]}, expected 1'
    assert obs_dir.shape[0] == 1, \
        f'Observation dir shape[0]={obs_dir.shape[0]}, expected 1'

    spacial_factor = (kappa/np.pi
        * np.exp(-kappa
                * sqeuclidean(source_dir - obs_dir)/2
                )
            ) # bivariate normal with kappa = 1/sigma^2
    flux_weight_factor = omega_s(0)
    detector_acc_factor = omega_acc(0)
    return spacial_factor * flux_weight_factor * detector_acc_factor


def bg_likelihood(obs_dir: np.ndarray):
    """ If Uniform, normalisation doesn't matter since the TS is relative
    """
    return 1


def test_stat(source_dir: np.ndarray, obs_dir: np.ndarray, kappa: float):
    N_obs = obs_dir.shape[0]
    TS = np.empty(N_obs)
    TS[:] = np.NaN
    for j in range(N_obs):
        S = 0
        for i in range(source_dir.shape[0]):
            S += source_likelihood(source_dir[i:i+1,:], obs_dir[j:j+1,:], kappa)
        TS[j] = 2*np.log(S / bg_likelihood(obs_dir[j:j+1,:]))
    return TS


def unit_vectors_skymap(unit_vector_dir: np.ndarray, labels: np.ndarray = None):
    if labels is not None:
        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == unit_vector_dir.shape[0]
    coords = SkyCoord(
            unit_vector_dir.T[0],
            unit_vector_dir.T[1],
            unit_vector_dir.T[2],
            representation_type="cartesian",
            frame="icrs",
        )
    coords.representation_type = "spherical"

    fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
    fig.set_size_inches((7, 5))
    if labels is None:
        for ra, dec in zip(
            coords.icrs.ra,
            coords.icrs.dec,
        ):
            circle = SphericalCircle(
                (ra, dec),
                3 * u.deg,
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )
            ax.add_patch(circle)
    else:
        label_cmap = plt.cm.tab10(list(range(2)))
        for ra, dec, l in zip(
            coords.icrs.ra,
            coords.icrs.dec,
            labels,
        ):
            circle = SphericalCircle(
                (ra, dec),
                3 * u.deg,
                color=label_cmap[l],
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )
            ax.add_patch(circle)
    return coords, labels
