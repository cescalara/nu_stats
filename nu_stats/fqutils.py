import numpy as np


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
