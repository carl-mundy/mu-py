from astropy.cosmology.core import FLRW
import astropy.units as u
import numpy as np


from . import Galaxy


class Pair(object):

    def __init__(self, primary: Galaxy, secondary: Galaxy, cosmology):

        self.primary = primary
        self.secondary = secondary

        self._cosmology = cosmology

    def __repr__(self):
        return 'Pair({}, {})'.format(self.primary, self.secondary)

    @property
    def pair_pdf(self):
        """Calculate the combined redshift probability function"""

        numerator = self.primary.pz * self.secondary.pz
        denominator = 0.5 * (self.primary.pz + self.secondary.pz)

        return numerator / denominator

    def theta_mask(self, sep_min: u.Quantity, sep_max: u.Quantity) -> np.ndarray:
        """Calculate the angular separation mask between two galaxies based on their
        separation and a high and low separation limit

        Args:
            sep_min (u.Quantity): minimum separation
            sep_max (u.Quantity): maximum separation

        Returns:
            np.array[bool]
        """

        # Calculate theta_min for each redshift
        theta_min = self.calc_angular_separations(sep_min, self.primary.zgrid, self._cosmology)
        theta_max = self.calc_angular_separations(sep_max, self.primary.zgrid, self._cosmology)

        # Local copy of separation
        _sep = self.separation.to(u.radian)

        return np.logical_and(theta_min <= _sep, theta_max >= _sep)

    @staticmethod
    def calc_angular_separations(sep: u.Quantity, z: float, cosmology: FLRW) -> u.Quantity:
        """Calculate the minimum and maximum angular separation based on sky-plane
        separation range and redshift range

        Args:
            sep
            z
            cosmology

        Returns:
            float
        """

        return u.radian * sep.to(u.Mpc) / cosmology.angular_diameter_distance(z)

    @property
    def separation(self):

        return self.primary.coord.separation(self.secondary.coord).to(u.radian)


