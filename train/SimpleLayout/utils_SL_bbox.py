import numpy as np
from utils_SL_geo import isect_line_plane_v3

def get_corners_of_bb3d(basis, coeffs, centroid):
    '''
    coeffs are HALF edge lengths.
    Example:
        basis = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        coeffs = [2., 3., 1.] # half length
        centroid = np.array([coeffs[0], coeffs[1], coeffs[2]]).reshape((1, 3))
    # 
    '''
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]

    corners[4, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[6, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners

