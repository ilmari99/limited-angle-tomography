import sys
import os

import numpy as np
from skimage.segmentation import flood_fill

if __name__ == "__main__":
    from AbsorptionMatrix import AbsorptionMatrix
else:
    from .AbsorptionMatrix import AbsorptionMatrix
from typing import Callable, SupportsIndex, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Bezier import get_bezier_curve, get_random_points
import matplotlib.pyplot as plt


def reshape_matrix_by_padding_and_cutting(matrix, required_shape):
    """ Pad the matrix with zeros equally from all sides, so that the matrix has the required shape.
    We want to retain the center of the matrix, so we add rows/cols equally to the top/bottom/left/right.
    """
    current_shape_center = np.mean(np.argwhere(matrix), axis=0)
    current_matrix_center = (matrix.shape[0]//2, matrix.shape[1]//2)
    #print(f"Current shape center: {current_shape_center}")
    #print(f"Matrix center: {current_matrix_center}")
    
    #print(f"Input matrix: {matrix.shape}")
    #print(f"Required shape: {required_shape}")
    matrix = cut_matrix_to(matrix, required_shape)
    #print(f"Cut matrix: {matrix.shape}")
    current_shape = matrix.shape
    diff_rows = required_shape[0] - current_shape[0]
    diff_cols = required_shape[1] - current_shape[1]
    pad_top = diff_rows // 2
    pad_bottom = diff_rows - pad_top
    pad_left = diff_cols // 2
    pad_right = diff_cols - pad_left
    #print(f"Pad top: {pad_top}, pad bottom: {pad_bottom}, pad left: {pad_left}, pad right: {pad_right}")
    padded_matrix = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    new_shape_center = np.mean(np.argwhere(padded_matrix), axis=0)
    new_matrix_center = (padded_matrix.shape[0]//2, padded_matrix.shape[1]//2)
    #print(f"New shape center: {new_shape_center}")
    #print(f"New matrix center: {new_matrix_center}")
    return padded_matrix

def cut_matrix_to(matrix, required_shape):
    """ If the matrix has more rows/cols than required_shape, remove the excess equally from the top/bottom/left/right.
    """
    num_extra_rows = matrix.shape[0] - required_shape[0]
    num_extra_cols = matrix.shape[1] - required_shape[1]
    cut_top = max(num_extra_rows // 2,0)
    cut_bottom = max(num_extra_rows - cut_top,0)
    cut_left = max(num_extra_cols // 2,0)
    cut_right = max(num_extra_cols - cut_left,0)
    
    #print(f"Num extra rows: {num_extra_rows}, num extra cols: {num_extra_cols}")
    #print(f"Cut top: {cut_top}, cut bottom: {cut_bottom}, cut left: {cut_left}, cut right: {cut_right}")
    cut_matrix = matrix[cut_top:matrix.shape[0]-cut_bottom, cut_left:matrix.shape[1]-cut_right]
    return cut_matrix


    


class FilledBezierCurve(AbsorptionMatrix):
    """ This class is an absorption matrix, where the absorption is a
    shape where the shape is a random continuous area.
    """
    def __init__(self, matrix_size, shape_size_frac = 0.5, discard_if_shape_frac_lt = 0.05, on_fail = "ignore", rad = 0.2, edgy=0, n_bezier_points = 5, **kwargs):
        # Create a matrix of zeros with 'shape_size_bounds'
        # And add a random contiguous shape (area of ones) to it.
        curve_x, curve_y, _ = get_bezier_curve(get_random_points(n=n_bezier_points), rad=rad, edgy=edgy)
        # Scale points
        curve_x = np.array(curve_x * shape_size_frac * matrix_size[0], dtype=np.int32)
        curve_y = np.array(curve_y * shape_size_frac * matrix_size[1], dtype=np.int32)
        matrix = np.zeros((int(shape_size_frac * matrix_size[0]), int(shape_size_frac * matrix_size[1])), dtype=np.float32)
        # Add the curve to the matrix
        for i in range(len(curve_x)):
            matrix[curve_x[i], curve_y[i]] = 1

        # Find the largest distance between two pixels
        max_distance = 0
        for i in range(len(curve_x)):
            for j in range(i+1,len(curve_x)):
                distance = np.sqrt((curve_x[i]-curve_x[j])**2 + (curve_y[i]-curve_y[j])**2)
                if distance > max_distance:
                    max_distance = distance

        minimum_required_side_length = np.ceil(max_distance + 1)
        #print(f"Minimum required side length: {minimum_required_side_length}")

        #if matrix_center[0] > mass_center[0]:
        # Add rows to the top to center the shape
        # Firstly, add one row/col to make the matrix divisible by 2
        if matrix.shape[0] % 2 == 0:
            matrix = np.vstack((np.zeros((1,matrix.shape[1]), dtype=np.float32), matrix))
        if matrix.shape[1] % 2 == 0:
            matrix = np.hstack((np.zeros((matrix.shape[0],1), dtype=np.float32), matrix))
        
        matrix_center = (matrix.shape[0]//2, matrix.shape[1]//2)
        mass_center = np.mean(np.argwhere(matrix), axis=0)
        mass_center = (round(mass_center[0]), round(mass_center[1]))
        mass_center_to_matrix_center = (mass_center[0] - matrix_center[0], mass_center[1] - matrix_center[1])
        
        
        #print(f"Matrix center: {matrix_center}")
        #print(f"Mass center to matrix center: {mass_center_to_matrix_center}")
        #print(f"Matrix center to mass center: {mass_center_to_matrix_center}")
        
        d_rows = matrix_center[0] - mass_center[0]
        # If the difference is positive, add 2*d_rows rows to the top
        if d_rows > 0:
            matrix = np.vstack((np.zeros((2*d_rows,matrix.shape[1]), dtype=np.float32), matrix))
        # If the difference is negative, add to bottom
        elif d_rows < 0:
            matrix = np.vstack((matrix, np.zeros((-2*d_rows,matrix.shape[1]), dtype=np.float32)))
        d_cols = matrix_center[1] - mass_center[1]
        # If the difference is positive, add 2*d_cols cols to the left
        if d_cols > 0:
            matrix = np.hstack((np.zeros((matrix.shape[0],2*d_cols), dtype=np.float32), matrix))
        # If the difference is negative, add to right
        elif d_cols < 0:
            matrix = np.hstack((matrix, np.zeros((matrix.shape[0],-2*d_cols), dtype=np.float32)))

        #print(f"Matrix after padding to center mass center: {matrix.shape}")
        
        matrix = reshape_matrix_by_padding_and_cutting(matrix, matrix_size)
        #print(f"Padded matrix: {matrix.shape}")
        
        super().__init__(matrix.shape, **kwargs)

        # Calculate the center of the shape
        non_zero_points = np.argwhere(matrix)
        # Find the mean of the non-zero points
        center = np.mean(non_zero_points, axis=0)
        center = (round(center[0]), round(center[1]))
        self.shape_center = center
        
        self.matrix = matrix

        self.matrix = flood_fill(self.matrix, self.shape_center, 1,connectivity=1)
        self. SUCCESS = True
        
        non_zero_points = np.argwhere(self.matrix)
        frac_ones = len(non_zero_points) / (self.matrix.shape[0] * self.matrix.shape[1])
        if frac_ones < discard_if_shape_frac_lt or frac_ones > 0.7 or (self.shape_center[0] - self.center[0] > 1) or (self.shape_center[1] - self.center[1] > 1):
            self.SUCCESS = False
            if on_fail == "ignore":
                pass
            elif on_fail == "print":
                print(f"Bezier shape filling failed. Fraction of ones: {frac_ones}")
                print(f"Shape center: {self.shape_center}, matrix center: {self.center}")
            elif on_fail == "raise":
                raise ValueError(f"Bezier shape filling failed. Fraction of ones: {frac_ones}")
            elif on_fail == "visualize":
                print(f"Bezier shape filling failed. Fraction of ones: {frac_ones}")
                print(f"Shape center: {self.shape_center}, matrix center: {self.center}")
                fig, ax = self.plot()
                # Put a marker on the center
                ax.plot(self.shape_center[1], self.shape_center[0], "ro", label = "Shape center")
                ax.plot(self.center[0], self.center[1], "bo", label = "Matrix center")
                ax.set_title("Filled Bezier curve (matrix)")
                ax.legend()
                self.animate_spinning()
                plt.show()
            else:
                raise ValueError(f"on_fail must be one of 'ignore', 'raise' or 'visualize'. Got {on_fail}")
        return
    
    @classmethod
    def from_matrix(cls,matrix, **kwargs):
        """ Creates a bezier curve from a matrix.
        """
        curve = cls(matrix.shape, **kwargs)
        curve.matrix = matrix
        return curve
    
    def add_shape(self):
        pass

if __name__ == "__main__":
    while True:
        FilledBezierCurve((100,100), shape_size_frac=0.8, rad=0.2, edgy=0, n_bezier_points=5)
        # Clear plt
        plt.cla()
        plt.close()