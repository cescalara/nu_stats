from typing import Counter
import numpy as np
from astropy.visualization.wcsaxes.patches import _rotate_polygon
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import astropy.units as u


class SphericalCircle(PathPatch):
    # created from the astropy.visualization.wcsaxes.patches.SphericalCircle class
    # changed to path from polygon to create disjointed parts
    # code from https://github.com/grburgess/pyipn
    """
    Create a patch representing a spherical circle - that is, a circle that is
    formed of all the points that are within a certain angle of the central
    coordinates on a sphere. Here we assume that latitude goes from -90 to +90
    This class is needed in cases where the user wants to add a circular patch
    to a celestial image, since otherwise the circle will be distorted, because
    a fixed interval in longitude corresponds to a different angle on the sky
    depending on the latitude.
    Parameters
    ----------
    center : tuple or `~astropy.units.Quantity`
        This can be either a tuple of two `~astropy.units.Quantity` objects, or
        a single `~astropy.units.Quantity` array with two elements.
    radius : `~astropy.units.Quantity`
        The radius of the circle
    resolution : int, optional
        The number of points that make up the circle - increase this to get a
        smoother circle.
    vertex_unit : `~astropy.units.Unit`
        The units in which the resulting polygon should be defined - this
        should match the unit that the transformation (e.g. the WCS
        transformation) expects as input.
    Notes
    -----
    Additional keyword arguments are passed to `~matplotlib.patches.Polygon`
    """

    def __init__(self, center, radius, resolution=100, vertex_unit=u.degree, **kwargs):

        # Extract longitude/latitude, either from a tuple of two quantities, or
        # a single 2-element Quantity.

        longitude, latitude = center

        # #longitude values restricted on domain of (-180,180]
        # if longitude.to_value(u.deg) > 180. :
        # 	longitude = -360. * u.deg + longitude.to(u.deg)

        # Start off by generating the circle around the North pole
        lon = np.linspace(0.0, 2 * np.pi, resolution + 1)[:-1] * u.radian
        lat = np.repeat(0.5 * np.pi - radius.to_value(u.radian), resolution) * u.radian

        lon, lat = _rotate_polygon(lon, lat, longitude, latitude)

        # Extract new longitude/latitude in the requested units
        lon = lon.to_value(vertex_unit)
        lat = lat.to_value(vertex_unit)
        # Create polygon vertices
        vertices = np.array([lon, lat]).transpose()

        # split path into two sections if circle crosses -180, 180 bounds
        codes = []
        last = (4000.4 * u.degree).to_value(
            vertex_unit
        )  # 400.4 is a random number large enough so first element is "MOVETO"
        for v in vertices:
            if np.absolute(v[0] - last) > (300 * u.degree).to_value(vertex_unit):
                codes.append(Path.MOVETO)
            else:
                codes.append(Path.LINETO)
            last = v[0]

        circle_path = Path(vertices, codes)

        super().__init__(circle_path, **kwargs)


def compute_xyz(ra, dec, radius=100):

    out = np.zeros((len(ra), 3))

    for i, (r, d) in enumerate(zip(ra, dec)):

        out[i, 0] = np.cos(d) * np.cos(r)
        out[i, 1] = np.cos(d) * np.sin(r)
        out[i, 2] = np.sin(d)

    return radius * out


def get_lon_lat(center, theta, radius=1, resolution=100):

    longitude, latitude = center

    # #longitude values restricted on domain of (-180,180]
    # if longitude.to_value(u.deg) > 180. :
    # 	longitude = -360. * u.deg + longitude.to(u.deg)

    # Start off by generating the circle around the North pole
    lon = np.linspace(0.0, 2 * np.pi, resolution + 1)[:-1] * u.radian
    # lat = np.repeat(0.5 * np.pi - radius.to_value(u.radian), resolution) * u.radian
    lat = np.repeat(0.5 * np.pi - theta.to_value(u.radian), resolution) * u.radian

    lon, lat = _rotate_polygon(lon, lat, longitude, latitude)

    return lon, lat


def get_3d_circle(center, theta, radius, resolution=100):

    lon, lat = get_lon_lat(center, theta, radius, resolution)

    return compute_xyz(lon, lat, radius)

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
    count = 0
    if labels is None:
        for ra, dec in zip(
            coords.icrs.ra,
            coords.icrs.dec,
        ):
            count += 1
            if count>= 1000: continue
            circle = SphericalCircle(
                (ra, dec),
                3 * u.deg,
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )
            ax.add_patch(circle)
    else:
        label_cmap = plt.cm.Set1(list(range(2)))
        for ra, dec, l in zip(
            coords.icrs.ra,
            coords.icrs.dec,
            labels,
        ):
            count += 1
            if count>= 1000: continue
            circle = SphericalCircle(
                (ra, dec),
                3 * u.deg,
                color=label_cmap[l],
                alpha=0.5,
                transform=ax.get_transform("icrs"),
            )
            ax.add_patch(circle)

    return coords, labels