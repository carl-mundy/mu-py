from typing import Sequence

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np

from . import Galaxy, Tree


class MuPy(object):

    _default_cosmo_kwargs = {
        'H0': 70,
        'Om0': 0.3
    }

    def __init__(self, galaxies: Sequence[Galaxy], cosmology=None):

        self.galaxies = galaxies
        self._cosmology = cosmology or FlatLambdaCDM(**self._default_cosmo_kwargs)

        self.tree = self._create_tree()

    def find_pairs(self, r_min: u.Quantity, r_max: u.Quantity, z_min: float, z_max: float) -> Sequence:
        """Find pairs of galaxies within a certain range of each other"""

        sep_min, sep_max = self._calc_angular_separations(r_min, r_max, z_min, z_max)

    def _calc_angular_separations(self, r_min: u.Quantity, r_max: u.Quantity, z_min, z_max):
        """Calculate the minimum and maximum angular separation based on sky-plane
        separation range and redshift range

        Args:
            r_min
        """

        theta_min = r_min / self._cosmology.angular_diameter_distance(z_max)
        theta_max = r_max / self._cosmology.angular_diameter_distance(z_min)

        return theta_min, theta_max

    def _create_tree(self) -> Tree:
        """Create the KDTree of galaxy positions

        Returns:
            Tree
        """

        # Get the raw ra and dec from the galaxy objects
        ra = [galaxy.ra for galaxy in self.galaxies]
        dec = [galaxy.dec for galaxy in self.galaxies]

        # Convert them to radians
        ra = [_ra.to(u.rad) for _ra in ra]
        dec = [_dec.to(u.rad) for _dec in dec]

        # Put into correct format and create the tree
        positions = np.array([ra, dec])
        tree = Tree(positions)

        return tree

    def _are_normalised(self) -> bool:
        """Check all the galaxies have normalised PDFs

        Returns:
            bool
        """

        return all([galaxy.is_normalised for galaxy in self.galaxies])
