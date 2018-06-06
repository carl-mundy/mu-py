from astropy import log
import astropy.units as u
import numpy as np

from . import Galaxy


class Pair(object):

    def __init__(self, primary: Galaxy, secondary: Galaxy):

        self.primary = primary
        self.secondary = secondary

        log.debug('pair instance created successfully')

    def __repr__(self):
        return 'Pair({}, {})'.format(self.primary, self.secondary)

    @property
    def ppf(self) -> np.ndarray:
        """Calculate and return the Pair Probability Function"""
        return self.pair_pdf

    @property
    def pair_pdf(self) -> np.ndarray:
        """Calculate the combined redshift probability function

        Returns:
            np.ndarray
        """

        log.debug('calculating pair PDF ({})'.format(self))

        numerator = self.primary.pz() * self.secondary.pz()
        denominator = 0.5 * (self.primary.pz() + self.secondary.pz())

        return numerator / denominator

    def pair_mask(self, pri_log_lim, sec_log_lim, max_mass, mass_completeness):
        """Calculate the pair mass selection mask.
        Defined in Equation 4, Mundy et al. (2017)

        Args:
            pri_log_lim
            sec_log_lim
            max_mass
            mass_completeness

        Returns:
            np.ndarray[bool]
        """

        log.debug('calculating the pair mass selection mask ({})'.format(self))

        pri_log_lim = np.maximum(pri_log_lim, mass_completeness)
        sec_log_lim = np.maximum(sec_log_lim, mass_completeness)

        pri_mask = np.logical_and(self.primary.mz >= pri_log_lim, pri_log_lim <= max_mass)
        sec_mask = self.secondary.mz >= sec_log_lim

        return np.logical_and(pri_mask, sec_mask)

    # def theta_mask(self, sep_min: u.Quantity, sep_max: u.Quantity) -> np.ndarray:
    #     """Calculate the angular separation mask between two galaxies based on their
    #     separation and a high and low separation limit
    #
    #     Args:
    #         sep_min (u.Quantity): minimum separation
    #         sep_max (u.Quantity): maximum separation
    #
    #     Returns:
    #         np.array[bool]
    #     """
    #
    #     # Calculate theta_min for each redshift
    #     theta_min = self.calc_angular_separations(sep_min, self.primary._zgrid, self._cosmology)
    #     theta_max = self.calc_angular_separations(sep_max, self.primary._zgrid, self._cosmology)
    #
    #     # Local copy of separation
    #     _sep = self.separation.to(u.radian)
    #
    #     return np.logical_and(theta_min <= _sep, theta_max >= _sep)


    @property
    def separation(self):

        return self.primary.coord.separation(self.secondary.coord).to(u.radian)


