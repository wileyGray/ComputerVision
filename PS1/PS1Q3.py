import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io



class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        ###### START CODE HERE ######
        self.img = io.imread("inputPS1Q3.jpg")
        ###### END CODE HERE ######
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        gray = None
        ###### START CODE HERE ######
        temp = rgb.copy().astype(np.double)
        gray = np.dot(temp[:, :, :], [.2989, .5870, .1140])
        gray = gray.astype(np.uint8)
        ###### END CODE HERE ######
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = None
        ###### START CODE HERE ######
        newImg = self.img.copy()
        r = newImg[:,:,0]
        g = newImg[:,:,1]

        #print(newImg[0,0,:])
        swapImg = np.zeros(newImg.shape)
        swapImg[:,:,0] = g
        swapImg[:,:,1] = r
        swapImg[:,:,2] = newImg[:,:,2]
        #print(swapImg)

        plt.imshow(swapImg/255)
        plt.show()

        ###### END CODE HERE ######
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        grayImg = None
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        #from matplotlib import pyplot as plt
        print(grayImg)
        plt.imshow(grayImg,cmap='gray' , interpolation='nearest')
        plt.show()
        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        negativeImg = None
        ###### START CODE HERE ######
        negativeImg = 255 - self.prob_3_2() 
        
        plt.imshow(negativeImg,cmap='gray' , interpolation='nearest')
        plt.show()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        mirrorImg = None
        ###### START CODE HERE ######
        grayscale = self.prob_3_2()
        mirrorImg = np.fliplr(grayscale)
        plt.imshow(mirrorImg, cmap='gray' , interpolation='nearest')
        plt.show()
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        avgImg = None
        ###### START CODE HERE ######
        avgImg = ((self.prob_3_2().astype(np.double) + self.prob_3_4().astype(np.double)) / 2).astype(np.uint8)
        plt.imshow(avgImg ,cmap='gray' , interpolation='nearest')
        plt.show()
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """
        noisyImg, noise = [None]*2
        ###### START CODE HERE ######
        grayscale = self.prob_3_2()
        y, x = grayscale.shape
        noise = np.random.randint(0, 255, size = y * x)
        noise = noise.reshape(y, x)
        noisyImg = grayscale.astype(np.double) + noise
        noisyImg = np.clip(noisyImg, 0, 255)

        #noisyImg = noisyImg.astype(np.uint8)
        plt.imshow(noisyImg, cmap='gray' , interpolation='nearest')
        plt.show()

        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()



