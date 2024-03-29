import sys
import numpy as np
import matplotlib.image as mpimg
import os
import gudhi
from functools import reduce

def compute_persistence_diagram(matrix,min_pers=0,i=5):

    """
        Given a matrix representing a nii image compute the persistence diagram by using the Gudhi library (link)

        :param matrix: matrix encoding the nii image
        :type matrix: np.array

        :param min_pers: minimum persistence interval to be included in the persistence diagram
        :type min_pers: Integer

        :returns: Persistence diagram encoded as a list of tuples [d,x,y]=p where

            * d: indicates the dimension of the d-cycle p

            * x: indicates the birth of p

            * y: indicates the death of p
    """
    #save the dimenions of the matrix
    dims = matrix.shape
    size = reduce(lambda x, y: x * y, dims, 1)

    #create the cubica complex from the image
    cubical_complex = gudhi.CubicalComplex(dimensions=dims,top_dimensional_cells=np.reshape(matrix.T,size))
    #compute the persistence diagram
    if i == 5:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        return np.array(map(lambda row: [row[1][0],row[1][1]], pd))
    else:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        pd = cubical_complex.persistence_intervals_in_dimension(i)
        return np.array(list(map(lambda row: [row[0],row[1]], pd)))

def betti_number(imagely):
	# imagely_copy = mpimg.imread('output_IMG_1.png')
	imagely = imagely.detach().cpu().clone().numpy()
	width,height = imagely.shape
	imagely[width - 1, :] = 0
	imagely[:, height - 1] = 0
	imagely[0, :] = 0
	imagely[:, 0] = 0
	temp = compute_persistence_diagram(imagely, i = 1)
	betti_number = len(temp)
	# print (betti_number)
	return betti_number