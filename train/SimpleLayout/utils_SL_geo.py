import numpy as np
import cv2
import torch

def mask_for_polygons(polygons, im_size):
    """
    Convert a polygon or multipolygon list back to
    an image mask ndarray
    https://michhar.github.io/masks_to_polygons_and_back/
    """
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def get_front_3d_line(x1x2, front_flags, cam_center, cam_frontaxis, if_debug=False, if_torch=False):
    assert len(x1x2.shape)==2 and x1x2.shape[1]==3
    assert isinstance(front_flags, list)
    assert len(front_flags) == 2
    if if_debug:
        print('front_flags', front_flags)
    if not all(front_flags):
        if not front_flags[0] and not front_flags[1]:
            return None, None
        x_isect = isect_line_plane_v3(x1x2[0], x1x2[1], cam_center, cam_frontaxis, epsilon=1e-6, if_torch=if_torch)
        if if_torch:
            x1x2 = torch.vstack((x1x2[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        else:
            x1x2 = np.vstack((x1x2[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        return x1x2, [False, True]
    return x1x2, [False, False]

# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6, if_torch=False):
    """
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        output = add_v3v3(p0, u)
        if if_torch:
            return torch.stack(output).reshape((3, 1))
        else:
            return np.stack(output).reshape((3, 1))
    else:
        # The segment is parallel to plane.
        return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )
