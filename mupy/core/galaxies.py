from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


class Galaxy(object):

    _norm_tolerance = 1e-4

    def __init__(self,
                 uid: int,
                 ra: u.Quantity,
                 dec: u.Quantity,
                 redshift,
                 mass,
                 zgrid: np.array) -> None:

        self.uid = uid
        self.coord = SkyCoord(ra, dec)
        self.pz = redshift
        self.mz = mass
        self.zgrid = zgrid

    def __repr__(self) -> str:
        """Define how a galaxy is represented when printed

        Returns:
            str
        """

        return 'Galaxy({uid}, {ra:.4f}, {dec:.4f})'.format(
            uid=self.uid,
            ra=self.coord.ra.to(u.degree),
            dec=self.coord.dec.to(u.degree)
        )

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

        return abs(1 - self._integrate(self.zgrid, self.pz)) <= self._norm_tolerance

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
