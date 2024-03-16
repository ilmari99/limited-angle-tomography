import warnings
from .AbsorptionMatrix import AbsorptionMatrix
import numpy as np
import random
import math
from typing import Callable, SupportsIndex
from functools import lru_cache

class Full(AbsorptionMatrix):
    def __init__(self, **kwargs):
        """ Create a circle of absorption.
        """
        #super creates the matrix, and adds absorption to it using the add_absorption method.
        super().__init__(**kwargs)
    
    def add_shape(self):
        """ Fill the matrix with 1's
        """
        self.matrix = np.full(self.size,1,dtype=float)
        return self.matrix