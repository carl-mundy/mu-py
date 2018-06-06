from typing import Callable

from astropy import log
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree


class Field(object):

    def __init__(self, name: str, completeness_mass: np.ndarray, z_grid: np.ndarray):

        self.name = name
        self._completeness_fn = self._make_interpolation_fn(z_grid, completeness_mass)

        self.galaxies = []
        self.pairs = []

        log.debug('field instance created successfully')

    def __repr__(self):
        return 'Field({}, {} galaxies)'.format(self.name, len(self.galaxies))

    @staticmethod
    def _make_interpolation_fn(x: np.ndarray, y: np.ndarray) -> Callable:

        log.debug('creating interpolation function')
        fn = interp1d(
            x=x,
            y=y,
            bounds_error=True,
            kind='linear'
        )

        return fn

    def mass_completeness(self, z_grid: np.ndarray) -> np.ndarray:
        """Calculate and return the mass completeness on a specific redshift grid

        Args:
            z_grid (array-like): redshift grid to cast too

        Returns:
            np.ndarray
        """

        log.debug('interpolating mass completeness to input redshift grid')
        return self._completeness_fn(z_grid)

    def _to_kdtree(self) -> KDTree:
        """Convert the internal galaxy list to a kdtree

        Returns:
            KDTree
        """

        log.debug('generating KDTree from galaxy data')
        tree_data = [[galaxy.coord.ra.radian, galaxy.coord.dec.radian] for galaxy in self.galaxies]
        tree_data = np.array(tree_data)

        return KDTree(tree_data)
