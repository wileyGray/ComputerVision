import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


XAXIS = 1
YAXIS = 0

def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used later (in get_interest_points for calculating
    image gradients and a second moment matrix).
    You can call this function to get the 2D gaussian filter.
    
    Hints:
    1) Make sure the value sum to 1
    2) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################
    
    kernel = cv2.getGaussianKernel(ksize,sigma)
    kernel = kernel.dot(kernel.T)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filter, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Hints:
        Padding width should be half of the filter's shape (correspondingly)
        The conv_image shape should be same as the input image
        Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or colored (your choice)
    -   filter: filter that will be used in the convolution with shape (a,b)

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################
    #print(image.shape)
    filter      = filter[::-1,::-1]
    len         = filter.shape[YAXIS] - 1
    len_pad     = int(np.trunc(len / 2))
    padded_img  = cv2.copyMakeBorder(image, len_pad, len_pad, len_pad, len_pad, borderType = cv2.BORDER_CONSTANT, value = bias)
    #print(padded_img)
    i_scope     = padded_img.shape[YAXIS] - len
    j_scope     = padded_img.shape[XAXIS] - len

    #for index, x in np.ndenumerate(padded_img):
    #    print(index)
    conv_image  = np.zeros(image.shape)
    for i in range(i_scope):
        #print(i)
        for j in range(j_scope):
            #print(j)
            i_max            = i + len + 1
            j_max            = j + len + 1
            #print(conv_image.shape)
            #print("i_up_to:" + str(i_up_to))
            convolve         = padded_img[i: i_max, j: j_max]
            convolve         = convolve * filter
            conv_image[i, j] = np.sum(convolve)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               
    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################
    # https://en.wikipedia.org/wiki/Sobel_operator
    #[1, 0, -1], [2, 0, -2], [1, 0, -1]
    #[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]
    x       = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    y       = [[1, 2, 1],  [0, 0, 0],  [-1, -2, -1]]

    sobel_x = np.array(x)
    sobel_y = np.array(y)

    ix      = my_filter2D(image,sobel_x)
    iy      = my_filter2D(image,sobel_y)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################
    x_size      = int(x.shape[0])
    x_max       = 0
    y_max       = 0
    temp_x      = []
    temp_y      = []
    temp_c      = []
    N           = x_size
    size        = int(window_size / 2)
    
    if not window_size % 2:
        x_max  += 1
        y_max  += 1      

    x_max      += image.shape[YAXIS] - size
    y_max      += image.shape[XAXIS] - size

    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        c_val = c[i]
        if x_val >= size and x_val < x_max and y_val >= size and y_val < y_max:
            temp_x.append(x_val)
            temp_y.append(y_val)
            temp_c.append(c_val)

    x = np.array(temp_x)
    y = np.array(temp_y)
    c = np.array(temp_c)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D, get_gaussian_kernel

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################   
    ix2  = ix * ix
    iy2  = iy * iy
    ixiy = ix * iy

    gk   = get_gaussian_kernel(ksize, sigma)
    
    sx2  = my_filter2D(ix2, gk)
    sy2  = my_filter2D(iy2, gk)
    sxsy = my_filter2D(ixiy, gk)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments function below, calculate corner resposne.

    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################
    M     = [[sx2,sxsy] , [sxsy,sy2]]
    #det    = np.linalg.det(M)
    det   = sx2 * sy2 - sxsy * sxsy
    trace = np.trace(M)
    R     = det - trace * trace * alpha

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. 
    Take a matrix and return a matrix of the same size but only the max values in a neighborhood that are not zero. 
    We also do not want very small local maxima so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   neighborhood_size: int, the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################
    R_local_pts      = np.zeros(R.shape)
    mid              = np.median(R)
    R                = np.where(R < mid, 0, R)
    filter           = maximum_filter(R, size = neighborhood_size)
    R_local_pts      = np.where(R == filter, R, 0)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    By default, you do not need to make scale and orientation invariant to
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression. Once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Helpful function:
        get_gradients, second_moments, corner_response, non_max_suppression, remove_border_vals

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer, number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """

    x, y, R_local_pts, confidences = None, None, None, None

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################
    alpha          = .05
    ix, iy         = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R              = corner_response(sx2, sy2, sxsy, alpha)
    R_local_pts    = non_max_suppression(R)
    idxs           = np.where(R_local_pts != 0)
    x              = idxs[0]
    y              = idxs[1]
    confidences    = R_local_pts.flatten()
    idxs           = np.where(confidences != 0)
    confidences    = confidences[idxs]
    x, y, c        = remove_border_vals(image, x, y, confidences)
    n_pts          = np.minimum(len(x), n_pts)
    c_sort         = np.sort(c, axis = None)[len(c) - n_pts:]
    idxs           = np.where(np.in1d(c, c_sort))[0]
    confidences    = c[idxs]
    temp           = x[idxs]
    x              = y[idxs]
    y              = temp
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x,y, R_local_pts, confidences


