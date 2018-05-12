from typing import Sequence

from scipy.spatial import KDTree

from .galaxies import Galaxy


class Sky(object):

    def __init__(self):

        self._tree = None
        self._galaxies = []

    def add(self, *galaxies: Sequence[Galaxy]):