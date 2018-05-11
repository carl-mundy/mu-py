from astropy.units import Quantity
import numpy as np


class Galaxy(object):

    def __init__(self,
                 uid: int,
                 ra: Quantity,
                 dec: Quantity,
                 redshift,
                 zgrid: np.array) -> None:

        self.uid = uid
        self.ra = ra
        self.dec = dec
        self.pz = redshift
        self.zgrid = zgrid

    def prob(self, zlow: [int, float], zhigh: [int, float]) -> float:
        """Calculate the probability of the galaxy residing in the redshift range
        [zlow, zhigh).


        Args:
            zlow (float): redshift range lower limit
            zhigh (float): redshift range higher limit

        Returns:
            float
        """

        z_mask = np.logical_and(self.zgrid >= zlow, self.zgrid < zhigh)
        integral = self._integrate(self.zgrid[z_mask], self.pz[z_mask])

        return integral

    @property
    def is_normalised(self) -> bool:
        """Determine whether the galaxy's probability distribution is normalised

        Returns:
            bool
        """
        return abs(1 - self._integrate(self.zgrid, self.pz)) < 1e-4

    @staticmethod
    def _integrate(x: np.array, y: np.array) -> float:
        """Perform the integration under y given x

        Args:
            x (array-like)
            y (array-like)

        Returns:
            float
        """
        return np.trapz(y, x)
