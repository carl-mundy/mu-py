from typing import NoReturn

import numpy as np
from scipy.spatial import KDTree


class Tree(object):

    def __init__(self, tree_input: np.array) -> NoReturn:

        self._tree = KDTree(tree_input)
