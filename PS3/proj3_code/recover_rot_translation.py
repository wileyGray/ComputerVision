from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def recover_E_from_F(f_matrix: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    '''
    Recover the essential matrix from the fundamental matrix

    Args:
    -   f_matrix: fundamental matrix as a numpy array (shape=(3,3))
    -   k_matrix: the intrinsic matrix shared between the two cameras (shape=(3,3))
    Returns:
    -   e_matrix: the essential matrix as a numpy array (shape=(3,3))
    '''

    e_matrix = None

    ##############################
    e_matrix = np.dot(np.dot(k_matrix.T, f_matrix), k_matrix)
    ##############################

    return e_matrix

def recover_rot_translation_from_E(e_matrix: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    '''
    Decompose the essential matrix to get rotation and translation (upto a scale)

    Ref: Section 9.6.2 of Hartley and Zisserman's Multiple View Geometry in Computer Vision (Second Edition)

    Hint:
    - Refer to the docs for `Rotation.from_matrix` and `Rotation.as_rotvec` in scipy.spatial.transform module
    
    Args:
    -   e_matrix: the essential matrix as a numpy array (3 x 3 ndarray)
    Returns:
    -   R1: the (3,) array containing the rotation angles in radians; one of the two possible
    -   R2: the (3,) array containing the rotation angles in radians; other of the two possible
    -   t: a (3,) translation matrix with unit norm and +ve x-coordinate; if x-coordinate is zero then y should be positive, and so on.


    '''

    R1 = None
    R2 = None
    t = None

    ############################## 
    W    = np.array([[0, -1, 0], 
                     [1, 0, 0], 
                     [0, 0, 1]])
    u, _, v = np.linalg.svd(e_matrix, full_matrices = True)
    t    = u[:, 2]
    
    zeros = np.where(t != 0, 1, 0)
    #print(t[0])
    #print(t[1])
    #print(t[2])
    if t[0] < 0 or zeros[0] and t[1] < 0 or zeros[1] and t[2] < 0:
        t = -t
    dw   = np.dot(u, W)
    dwt  = np.dot(u, W.T)
    dwv  = np.dot(dw, v)
    dwvt = np.dot(dwt, v)
    R1   = Rotation.from_matrix(dwv).as_rotvec()
    R2   = Rotation.from_matrix(dwvt).as_rotvec()
    ##############################

    return R1, R2, t
