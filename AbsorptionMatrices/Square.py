import warnings
from .AbsorptionMatrix import AbsorptionMatrix
import numpy as np
import random
import math
from typing import Callable, SupportsIndex

class Square(AbsorptionMatrix):
    def __init__(self, side_len, **kwargs):
        """ Create a circle of absorption.
        """
        self.side_len = side_len
        # The size has to be such, that square can rotate around its center without losing any pixels.
        # So each side of the matrix has to have sqrt(2)*side_len as the size.
        sz = int(np.ceil(np.sqrt(2)*side_len))
        symmetry_angles = [0,90,180,270,360]
        super().__init__(size = [sz,sz], **kwargs)
    
    @classmethod
    def from_matrix(cls,matrix):
        """ Creates a circle from a matrix.
        """
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        # Intialize the circle, but override the attributes with the matrix
        side_len = matrix.shape[0]//np.sqrt(2)
        circle = cls(side_len)
        circle.matrix = matrix
        circle.size = matrix.shape
        return circle
    
    def add_shape(self):
        """ Add a square with side_len's
        """
        mat = self.matrix
        highest_row = int(self.center[0] - self.side_len//2)
        lowest_row = int(self.center[0] + self.side_len//2)
        leftmost_col = int(self.center[1] - self.side_len//2)
        rightmost_col = int(self.center[1] + self.side_len//2)
        print(f"highest_row: {highest_row}, lowest_row: {lowest_row}, leftmost_col: {leftmost_col}, rightmost_col: {rightmost_col}")
        # Fill the square with 1's
        mat[highest_row:lowest_row,leftmost_col:rightmost_col] = 1
        return mat
    
if __name__ == "__main__":
    sq = Square(256, rotate_kwargs = {"mode" : "constant", "cval" : 0})
    fig,ax = sq.plot(show=False, block=False)
    sq.rotate(45, inplace=True)
    sq.make_holes(6, 0.4, inplace=True)
    fig,ax = sq.plot(show=False, block=False)
    sq.rotate(45, inplace=True)
    fig,ax = sq.plot(show=True, block=True)
    #fig,ax = sq.plot(show=True, block=True)
    
        