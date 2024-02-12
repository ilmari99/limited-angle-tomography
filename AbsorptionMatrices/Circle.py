import warnings
from .AbsorptionMatrix import AbsorptionMatrix
import numpy as np
import random
import math
from typing import Callable, SupportsIndex
from functools import lru_cache

class Circle(AbsorptionMatrix):
    def __init__(self, radius, **kwargs):
        """ Create a circle of absorption.
        """
        self.radius = radius
        #super creates the matrix, and adds absorption to it using the add_absorption method.
        super().__init__(size = [2*radius + 1, 2*radius + 1], **kwargs)
        self.center = (self.size[0]//2,self.size[1]//2)
    
    @classmethod
    def from_matrix(cls,matrix):
        """ Creates a circle from a matrix.
        """
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        # Intialize the circle, but override the attributes with the matrix
        circle = cls(1)
        circle.matrix = matrix
        circle.size = matrix.shape
        return circle
    
    def add_shape(self):
        """ Add a circle of absorption to the matrix.
        """
        self.matrix = add_circle_to_matrix(self.matrix,self.center,self.radius)
        return self.matrix
    
    def is_inside_circle(self,point):
        """ Returns True if the point is inside the circle, False otherwise.
        """
        return is_inside_circle(point,self.center,self.radius) 

ADD_CIRCLE_TO_MATRIX_CACHE = {}
def add_circle_to_matrix(matrix, center, radius):
    """ Add a circle of absorption to the matrix.
    """
    if (matrix.shape,center,radius) in ADD_CIRCLE_TO_MATRIX_CACHE:
        return ADD_CIRCLE_TO_MATRIX_CACHE[(matrix.shape,center,radius)]
    # Create a circle with radius self.radius
    mat = matrix
    # Add a filled circle to the matrix
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            if is_inside_circle((row,col), center, radius):
                mat[row,col] = 1
    ADD_CIRCLE_TO_MATRIX_CACHE[(matrix.shape,center,radius)] = mat
    return mat

def is_inside_circle(point,center, radius):
    """ Returns True if the point is inside the circle, False otherwise.
    """
    # Get the distance from the center of the circle to the point
    dist = math.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
    # If the distance is less than the radius, the point is inside the circle
    return dist <= radius