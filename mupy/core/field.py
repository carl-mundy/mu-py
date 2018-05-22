from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


class Field(object):

    def __init__(self, name: str, completeness_mass: np.ndarray, z_grid: np.ndarray):

        self.name = name
        self._completeness_fn = self._make_interpolation_fn(z_grid, completeness_mass)

        self.galaxies = []

    @staticmethod
    def _make_interpolation_fn(x: np.ndarray, y: np.ndarray) -> Callable:

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

        return self._completeness_fn(z_grid)
