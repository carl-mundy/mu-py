from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d


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

        self._pz = redshift
        self.mz = mass
        self._zgrid = zgrid

        self._pz_interp = interp1d(y=self._pz, x=self._zgrid)

    def w1(self, zgrid=None):

        if zgrid is None:
            zgrid = self._zgrid

        return np.ones(zgrid.shape, dtype=bool)

    def pz(self, zgrid=None):

        if zgrid is None:
            zgrid = self._zgrid

        return self._pz_interp(zgrid)

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

        z_mask = np.logical_and(self._zgrid >= zlow, self._zgrid < zhigh)
        integral = self._integrate(self._zgrid[z_mask], self.pz()[z_mask])

        return integral

    @property
    def is_normalised(self) -> bool:
        """Determine whether the galaxy's probability distribution is normalised

        Returns:
            bool
        """

        return abs(1 - self._integrate(self._zgrid, self.pz())) <= self._norm_tolerance

    @staticmethod
    def _integrate(x: np.array, y: np.array) -> float:
        """Perform the integration under y given x

        Args:
            x (array-like)
            y (array-like)

        Returns:
            float
        """
        return simps(y=y, x=x)
