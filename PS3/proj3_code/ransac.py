import math
from typing import Tuple

import numpy as np

from proj3_code import fundamental_matrix, two_view_data
from proj3_code.least_squares_fundamental_matrix import solve_F


def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int representing the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    #https://helios2.mi.parisdescartes.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf
    #num_samples = k = log(prob failure) / log(prob all elements fail)

    prob_failure     = 1 - prob_success
    prob_failure_ind = 1 - (ind_prob_correct ** sample_size)
    num_samples      = int(np.log(prob_failure) / np.log(prob_failure_ind))
    ##############################

    return num_samples


def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix of shape (3, 3)
    -   x_1s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: float representing the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """
    inliers = []

    ##############################
    N, M  = x_0s.shape
    shape = (N, 3)
    
    new0  = np.ones(shape)
    new1  = np.ones(shape)
    size  = len(new0)

    if M == 3:
        new0 = x_0s
        new1 = x_1s
    else:
        new0[:, :-1] = x_0s
        new1[:, :-1] = x_1s
    
    for idx in range(size):
        line  = np.dot(F, new1[idx])
        point = new0[idx]
        dist  = np.abs(fundamental_matrix.point_line_distance(line, point))
        diff  = threshold - dist
        if diff >= 0:
            inliers.append(idx)

    inliers = np.array(inliers)
    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.
        4. You can use the `preprocess_data` function in `two_view_data` to make
           x_0s, and x_1s homogeneous.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None

    ##############################
    #v = np.zeros
    M                = int(x_0s.shape[0])
    #print(M)
    F_shape          = (3, 3)
    precision        = 100
    P                = .99
    k                = 7
    p                = .7
    threshold        = 1
    iterations       = calculate_num_ransac_iterations(P, k, p)
    print("num of req iterations: " + str(iterations))
    old_inliers      = []
    best_F           = 0

    #training
    for i in range(iterations + 1):

        if(i == iterations - 1):
            break

        rand_idxs        = np.random.choice(M, k)
        temp_F           = solve_F(x_0s[rand_idxs], x_1s[rand_idxs])
        new_inliers      = find_inliers(x_0s, temp_F, x_1s, threshold)
        num_old          = len(old_inliers)
        num_new          = len(new_inliers)

        if num_old < num_new:
            old_inliers  = new_inliers
            best_F       = temp_F

    inlier_idxs  = find_inliers(x_0s, best_F, x_1s, threshold)
    inliers_x_0  = x_0s[inlier_idxs]
    inliers_x_1  = x_1s[inlier_idxs]
    ##############################

    return best_F, inliers_x_0, inliers_x_1
