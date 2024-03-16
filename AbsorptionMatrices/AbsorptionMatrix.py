
from abc import ABC, abstractmethod
import random
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from typing import Callable
import numpy as np

class AbsorptionMatrix(ABC):

    def __init__(self, size, seed = None, rotate_kwargs = {}):
        random.seed(seed)
        np.random.seed(seed)
        #print(f"Creating a matrix of size {size}")
        #assert isinstance(size,SupportsIndex), "size must support indexing."
        assert size[0] == size[1], "Absorption matrix must be a square."
        assert len(size) == 2, "Only 2D arrays are supported"
        self.size = size
        self.rotate_kwargs = rotate_kwargs
        self.center = (self.size[0]//2,self.size[1]//2)
        self.matrix = self.init_matrix(size=size)
        self.matrix = self.add_shape()
        self.rotation_degree = 0
    
    @abstractmethod
    def add_shape(self):
        """ This method adds absorption to self.matrix, by generating tuples of (row,col,absorption)
        """
        pass
    
    @classmethod
    def from_matrix(cls, matrix, rotate_kwargs = {}):
        """ Create an AbsorptionMatrix from a matrix.
        """
        size = matrix.shape
        instance = cls(size, rotate_kwargs=rotate_kwargs)
        instance.matrix = matrix
        return instance
        
    
    def init_matrix(self, size):
        matrix = np.full(size,0,dtype=float)
        return matrix
    
    def rotate(self,angle, spline_order=5, inplace=False):
        mat = self.matrix if inplace else self.matrix.copy()
        default_kwargs = {"reshape":False, "order":spline_order}
        # default_kwargs take precedence over self.rotate_kwargs
        rotate_kwargs = {**default_kwargs, **self.rotate_kwargs}
        mat = rotate(mat,angle,**rotate_kwargs)
        if inplace:
            self.rotation_degree += angle
            self.matrix = mat
        return mat
    
    def plot(self, show=False, block=False):
        fig,ax = plt.subplots()
        ax.matshow(self.matrix)
        ax.set_title("rotated by {} degrees".format(self.rotation_degree))
        if show:
            plt.show(block = block)
        return fig,ax
    
    def init_absorption(self,absorption):
        if isinstance(absorption,(int,float)):
            self.absorption = lambda : absorption
        elif isinstance(absorption,Callable):
            self.absorption = absorption
        else:
            raise TypeError("Absorption must be an integer, float or a callable distribution.") 
    
    def make_regular_holes(self,hole_sz, n_holes, inplace = True):
        """ Makes n_holes of size hole_sz to the circle in random locations where absorption is not 0.
        So sets the value of the hole to 0.
        """
        mat = self.matrix if inplace else self.matrix.copy()
        for ith_hole in range(n_holes):
            hole_lu_corner = (random.randint(0,self.size[0]-hole_sz[0]),random.randint(0,self.size[1]-hole_sz[1]))
            # Set the values at indices to 0
            mat[hole_lu_corner[0]:hole_lu_corner[0]+hole_sz[0],hole_lu_corner[1]:hole_lu_corner[1]+hole_sz[1]] = 0
        return mat
    
    def make_holes(self, n, n_missing_pixels, hole_size_volatility = 0.5, hole_ratio_limit = 10, at_angle = 0, inplace = True):
        """ Make n random size holes holes in the circle, so that the total number of missing pixels is n_missing_pixels.
        """
        n_missing_pixels = n_missing_pixels * len(self.matrix[self.matrix > 0]) if n_missing_pixels < 1 else n_missing_pixels
        mat = self.rotate(at_angle, inplace=inplace)
        curr_missing_px = 0
        still_missing_px = n_missing_pixels - curr_missing_px
        base_hole_sz = n_missing_pixels / n
        for nth_hole in range(n):
            # The base hole size is when all holes are the same size
            base_hole_size = min(base_hole_sz,still_missing_px)
            # The hole size volatility defines how much the hole size can vary from the average hole size.
            # If hole_size_volatility = 0, then the hole size is always the same: n_missing_pixels / n
            # If hole_size_volatility = 1, then the hole size can be anything from 0 to n_missing_pixels
            # How much larger the hole size can be than the base hole size
            hole_size_change_upper = hole_size_volatility * (n_missing_pixels - base_hole_size)
            # How much smaller the hole size can be than the base hole size
            hole_size_change_lower = -hole_size_volatility * base_hole_size
            smallest_hole_size_px = max(base_hole_size + hole_size_change_lower, 1)
            largest_hole_size_px = min(base_hole_size + hole_size_change_upper + 1, still_missing_px)
            # The area that a hole takes up is h*w, and
            # smallest_hole_size_px <= h*w <= largest_hole_size_px
            # Additionally 0.1 <= h/w <= 10
            # So, we can choose a random hole size
            #print(f"Choosing a hole size (mean {base_hole_size}, min {smallest_hole_size_px}, max {largest_hole_size_px})")
            chosen_hole_size_px = np.random.randint(smallest_hole_size_px,min(largest_hole_size_px,smallest_hole_size_px)+1)
            possible_hs = [i for i in range(1,int(np.sqrt(chosen_hole_size_px))+1)]
            random.shuffle(possible_hs)
            
            # Try 10 different hs
            for h in possible_hs:
                w = chosen_hole_size_px // h
                if 1/hole_ratio_limit <= h/w <= hole_ratio_limit:
                    break
            else:
                raise ValueError("Could not find a valid hole size.")
            # Now we have a valid hole size
            #hole_lu_corner = (random.randint(0,self.size[0]-h),random.randint(0,self.size[1]-w))
            # Select a random starting point, that equals == 1
            hole_lu_corner = random.choice(np.argwhere(mat > 0))
            mat[hole_lu_corner[0]:hole_lu_corner[0]+h,hole_lu_corner[1]:hole_lu_corner[1]+w] = 0
            still_missing_px -= chosen_hole_size_px
            #print(f"Created hole with size {h}x{w} using {chosen_hole_size_px} pixels. Still missing {still_missing_px} pixels.")
            if still_missing_px <= 0.01 * n_missing_pixels:
                break
        if inplace:
            self.rotate(-at_angle, inplace=True)
        return mat
    
    def get_multiple_measurements(self,angles, spline_order=5, add_noise = lambda x : x, return_distances = False, zero_threshold = 0.01):
        measurements = np.zeros((len(angles), self.matrix.shape[0]), dtype=np.float32)
        distances_from_front = np.zeros((len(angles), self.matrix.shape[0]), dtype=np.int32)
        distances_from_back = np.zeros((len(angles), self.matrix.shape[0]), dtype=np.int32)
        for i,angle in enumerate(angles):
            m, dfdb = self.get_measurement(angle, return_distances=return_distances, zero_threshold=zero_threshold, spline_order=spline_order, add_noise=add_noise)
            # df: distance from front, db: distance from back, m: amount of absorption at each height (in this case, the sum of the row)
            df = dfdb[0,:]
            db = dfdb[1,:]
            measurements[i,:] = m
            distances_from_front[i,:] = df
            distances_from_back[i,:] = db
        return measurements, distances_from_front, distances_from_back
        
    
    def get_measurement(self,theta : float,
                        spline_order=5,
                        add_noise = lambda x : x,
                        return_distances = False,
                        zero_threshold = 0.01
                        ):
        """ Returns the total absorption at each height of the circle in some angle theta.
        If return_distances is True, then we also return the distance to the surface of the object at each height, from front and back.
        """
        # Rotate the matrix
        rotated_mat = self.rotate(theta,spline_order=spline_order)
        # Get the row sums of the rotated matrix
        absorption = rotated_mat.sum(axis=1)
        # Add noise to the measurements
        absorption = add_noise(absorption)
        if return_distances:
            # Find the distance to the surface of the object at each height, from front and back
            distances = np.zeros((2,self.size[0]))
            # At each height, find the distance to the first pixel above 0.01
            for i in range(self.size[0]):
                df = np.argmax(rotated_mat[i,:] > zero_threshold)
                db = np.argmax(rotated_mat[i,::-1] > zero_threshold)
                
                # If the argmax is 0 and no pixel is above threshold, then we know
                # that there is no object at this height, so we set the distance to the middle
                if np.all(rotated_mat[i,:] <= zero_threshold):
                    #if i > 40 and i < 70:
                    #    print(f"No object at height {i} at angle {theta}.")
                    df = self.size[1]//2
                    db = self.size[1]//2
                distances[0,i] = df
                distances[1,i] = db
                
            distances = distances.astype(np.int32)
            return absorption, distances
        return absorption
    
    def animate_spinning(self):
        """ Creates a figure, where the matrix is continuously spinning 1 degree at a time.
        The animation is started only when plt.show() is called.
        """
        fig,ax = plt.subplots()
        ax.set_title("Spinning circle")
        for i in range(360):
            self.rotate(1, inplace=True)
            ax.matshow(self.matrix)
            plt.pause(0.01)
            ax.clear()
        plt.show()
    
    def __eq__(self,other):
        if isinstance(other, AbsorptionMatrix):
            return np.array_equal(self.matrix, other.matrix)
        return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.size})"
        
    # Make the class pickleable
    def ___reduce__(self):
        # from matrix
        return (self.__class__.from_matrix, (self.matrix, self.rotate_kwargs))
        
        