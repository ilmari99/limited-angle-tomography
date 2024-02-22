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



class FilledBezierCurve(AbsorptionMatrix):
    """ This class is an absorption matrix, where the absorption is a
    shape where the shape is a random continuous area.
    """
    def __init__(self, shape_size_bounds : Tuple[int,int], rad = 0.2, edgy=0, n_bezier_points = 12, **kwargs):
        # Create a matrix of zeros with 'shape_size_bounds'
        # And add a random contiguous shape (area of ones) to it.
        curve_x, curve_y, _ = get_bezier_curve(get_random_points(n=n_bezier_points), rad=rad, edgy=edgy)
        # Scale points
        curve_x = np.array(curve_x * shape_size_bounds[0], dtype=np.int32)
        curve_y = np.array(curve_y * shape_size_bounds[1], dtype=np.int32)
        matrix = np.zeros(shape_size_bounds, dtype=np.float32)
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

        print(f"Max distance: {max_distance}")
        side_length = int(max_distance + 1)
        self.size = (side_length, side_length)

        # Calculate the center of the shape
        non_zero_points = np.argwhere(matrix)
        # Find the mean of the non-zero points
        center = np.mean(non_zero_points, axis=0)
        mass_center = (round(center[0]), round(center[1]))
        print(f"Old center: {center}")
        print(f"Matrix shape: {matrix.shape}")
        
        matrix_center = (matrix.shape[0]//2, matrix.shape[1]//2)
        mass_center_to_matrix_center = (mass_center[0] - matrix_center[0], mass_center[1] - matrix_center[1])
        print(f"Matrix center: {matrix_center}")
        print(f"Mass center to matrix center: {mass_center_to_matrix_center}")
        print(f"Matrix center to mass center: {mass_center_to_matrix_center}")

        if matrix_center[0] > mass_center[0]:
            # Add rows to the top to center the shape
            # Firstly, add one row/col to make the matrix divisible by 2
            if matrix.shape[0] % 2 == 0:
                matrix = np.vstack((np.zeros((1,matrix.shape[1]), dtype=np.float32), matrix))
            if matrix.shape[1] % 2 == 0:
                matrix = np.hstack((np.zeros((matrix.shape[0],1), dtype=np.float32), matrix))
            d_rows = matrix_center[0] - mass_center[0]
            # If the difference is positive, add 2*d_rows rows to the top
            if d_rows > 0:
                matrix = np.vstack((np.zeros((2*d_rows,matrix.shape[1]), dtype=np.float32), matrix))
            # If the difference is negative, add to bot
            elif d_rows < 0:
                matrix = np.vstack((matrix, np.zeros((-2*d_rows,matrix.shape[1]), dtype=np.float32)))
            d_cols = matrix_center[1] - mass_center[1]
            # If the difference is positive, add 2*d_cols cols to the left
            if d_cols > 0:
                matrix = np.hstack((np.zeros((matrix.shape[0],2*d_cols), dtype=np.float32), matrix))
            # If the difference is negative, add to right
            elif d_cols < 0:
                matrix = np.hstack((matrix, np.zeros((matrix.shape[0],-2*d_cols), dtype=np.float32)))

        print(f"Padded matrix: {matrix.shape}")
        super().__init__(self.size, **kwargs)

        # Calculate the center of the shape
        non_zero_points = np.argwhere(matrix)
        # Find the mean of the non-zero points
        center = np.mean(non_zero_points, axis=0)
        center = (round(center[0]), round(center[1]))
        self.center = center
        print(f"New center: {center}")
        
        self.matrix = matrix

        # Fill from the center
        p = self.center
        print(f"Shape center: {p}")

        self.matrix = flood_fill(self.matrix, p, 1,connectivity=1)

        non_zero_points = np.argwhere(matrix)
        frac_ones = len(non_zero_points) / (matrix.shape[0] * matrix.shape[1])
        if frac_ones < 0.015 or frac_ones > 0.03:
            print(f"Bezier shape filling failed. Fraction of ones: {frac_ones}")
            fig, ax = self.plot()
            # Put a marker on the center
            ax.plot(p[1], p[0], "ro", label = "Shape center")
            ax.plot(matrix.shape[1]//2, matrix.shape[0]//2, "bo", label = "Matrix center")
            ax.set_title("Filled Bezier curve (matrix)")
            plt.show()
        #plt.show()
    
    def add_shape(self):
        pass

if __name__ == "__main__":
    while True:
        FilledBezierCurve((101,101))
        # Clear plt
        plt.cla()
        plt.close()