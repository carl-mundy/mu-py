from astropy import log
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
from scipy.integrate import simps

from . import Field, Pair, Galaxy


class MergerCase(object):

    # Define the default cosmology keyword arguments
    _default_cosmo_kwargs = {
        'Om0': 0.30,
        'H0': 70.0 * u.km / u.second / u.Mpc,
    }

    def __init__(self, min_sep: u.Quantity, max_sep: u.Quantity, z_grid: np.ndarray, cosmology=None):

        self.fields = []
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.cosmology = cosmology or FlatLambdaCDM(**self._default_cosmo_kwargs)

        self._zgrid = z_grid
        self._primary_selection_mask = np.ones(self._zgrid.shape, dtype=bool)

    def add_field(self, field: Field):
        self.fields.append(field)

    def process_field(self, field, z_min, z_max):
        """Process a Field by:

            1. Creating the KDTree of galaxy positions
            2. Finding pairs in the separation range
            3. Generating the pairs and setting them

        Args:
            field (Field)

        Returns:
        """

        # Convert to KDTree
        field_tree = field._to_kdtree()

        # Filter on separation conditions
        max_radians = self._phys_sep_2_angular_sep(self.max_sep, self._zgrid, self.cosmology).max()
        min_radians = self._phys_sep_2_angular_sep(self.min_sep, self._zgrid, self.cosmology).min()

        log.debug('querying tree for pairs')
        max_pairs = field_tree.query_pairs(max_radians.value)
        min_pairs = field_tree.query_pairs(min_radians.value)

        log.debug('found {} pairs below minimum separation limit'.format(len(min_pairs)))
        log.debug('found {} pairs below maximum separation limit'.format(len(max_pairs)))

        # Choose pairs only in the separation range
        unfiltered_pair_indices = max_pairs - min_pairs

        pairs = []
        for pair_indices in unfiltered_pair_indices:

            # Make the pair
            pair = self._make_pair(field, *pair_indices)
            log.debug('created pair ({})'.format(pair))

            pairs.append(pair)

        field.pairs = pairs

    @staticmethod
    def _determine_primary(*galaxies: Galaxy) -> int:
        """Determine which galaxy is the primary galaxy out of a sequence of input Galaxy instances

        Args:
            galaxies

        Returns:
            int
        """

        # What galaxy is more massive on average?
        mean_masses = [np.nanmean(galaxy.mz) for galaxy in galaxies]
        primary_index = mean_masses.index(max(mean_masses))

        return primary_index

    def _make_pair(self, field: Field, index1: int, index2: int):
        """Make a Pair instance from two galaxies

        Args:
            field (Field)
            index1 (int)
            index2 (int)

        Returns:
            Pair
        """

        # Retrieve references to the galaxies
        galaxy1 = field.galaxies[index1]
        galaxy2 = field.galaxies[index2]

        galaxies = [galaxy1, galaxy2]

        # Determine which should be primary and which should be secondary
        primary_index = self._determine_primary(galaxy1, galaxy2)

        # Create pair
        pair = Pair(
            primary=galaxies.pop(primary_index),
            secondary=galaxies[0]
        )

        return pair

    def calc_pair_fraction(self, z_min, z_max):
        """Calculate the pair fraction between two redshift limits

        Args:
            z_min (float)
            z_max (float)

        Returns:
            float
        """

        redshift_mask = np.logical_and(self._zgrid >= z_min, self._zgrid < z_max)

        field_pair_fractions = []
        for field in self.fields:

            # Find the pairs in a field
            self.process_field(field, z_min, z_max)

            # Calculate the pair fraction
            fpair = self._calc_fpair_in_field(field, z_min, z_max)

            field_pair_fractions.append(fpair)

        return field_pair_fractions

    def _calc_fpair_in_field(self, field, z_min, z_max):

        N_pair = self._calc_Npairs_in_field(field)
        N_primary = self._calc_Nprimary_in_field(field)

        return N_pair / N_primary

    def _calc_Nprimary_in_field(self, field: Field) -> float:
        """Calculate the number of primary galaxies within a field

        Args:
            field (Field)

        Returns:
            float
        """

        n_pri = 0

        for galaxy in field.galaxies:

            integrand = galaxy.pz(self._zgrid) * self._primary_selection_mask * galaxy.w1(self._zgrid)
            contrib = simps(y=integrand, x=self._zgrid)

            n_pri += contrib

        return n_pri

    def _calc_Npairs_in_field(self, field):
        """Calculate the number of pairs a sequence of pairs contribute

        Args:
            field: Field

        Returns:
            float
        """

        N_pair = 0

        for pair in field.pairs:
            contrib = simps(y=pair.ppf, x=self._zgrid)
            N_pair += contrib

        return N_pair

    @staticmethod
    def _phys_sep_2_angular_sep(sep: u.Quantity, z, cosmology) -> u.Quantity:
        """Calculate the minimum and maximum angular separation based on sky-plane
        separation range and redshift range

        Args:
            sep
            z
            cosmology

        Returns:
            float
        """

        sep_mpc = sep.to(u.Mpc)
        add = cosmology.angular_diameter_distance(z)

        radians = sep_mpc / add

        return radians
