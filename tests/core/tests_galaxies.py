import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import numpy.testing as nt

from mupy.core import Galaxy


class Tests_Galaxy(unittest.TestCase):

    _zgrid = np.arange(0, 3., 0.01)

    def _make_mock_pz(self, z_mid, std):
        return (1 / (std * (2 * np.pi)**(1/2))) * np.exp(-0.5 * ((self._zgrid - z_mid)/std)**2)

    def _make_mock_gal(self):
        uid = 1
        ra = 34 * u.deg
        dec = -5 * u.deg
        redshift = self._make_mock_pz(1., 0.1)
        mass = 11.1
        zgrid = self._zgrid

        galaxy = Galaxy(uid, ra, dec, redshift, mass, zgrid)

        return galaxy

    def test_init_properties(self):

        galaxy = self._make_mock_gal()

        self.assertEqual(galaxy.uid, 1)
        self.assertIsInstance(galaxy.coord, SkyCoord)
        self.assertEqual(galaxy.mz, 11.1)

        nt.assert_array_equal(galaxy.pz, self._make_mock_pz(1., 0.1))
        nt.assert_array_equal(galaxy._zgrid, self._zgrid)

    def test_galaxy_prob_is_normed(self):
        """Test that the galaxy probability equals 1 over the redshift range"""

        galaxy = self._make_mock_gal()

        self.assertTrue(galaxy.is_normalised)

    def test_galaxy_prob_is_1(self):

        galaxy = self._make_mock_gal()
        prob = galaxy.prob(0, 3.)
        self.assertAlmostEqual(prob, 1, 1e-4)

    def test_galaxy_prob_is_not_normed(self):
        """Test that the galaxy probability does over the redshift range"""

        galaxy = self._make_mock_gal()
        galaxy.pz = np.zeros_like(galaxy._zgrid)

        self.assertFalse(galaxy.is_normalised)

    def test_galaxy_prob_is_0(self):

        galaxy = self._make_mock_gal()
        galaxy.pz = np.zeros_like(galaxy._zgrid)

        prob = galaxy.prob(0, 3.)

        self.assertAlmostEqual(prob, 0, 1e-4)
