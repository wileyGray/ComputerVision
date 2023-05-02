import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        self.img = io.imread("inputPS1Q4.jpg")
        self.indoor = io.imread("indoor.png")
        self.outdoor = io.imread("outdoor.png")
        #plt.imshow(self.indoor, cmap='gray' , interpolation='nearest')
        #plt.show()
        #plt.imshow(self.outdoor, cmap='gray' , interpolation='nearest')
        #plt.show()
        ###### END CODE HERE ######

    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######

        print(self.indoor.shape)

        plt.imshow(self.indoor, cmap=plt.get_cmap('gray'))
        plt.title("indoor image")
        plt.show()

        ##indoor Red grayscale
        plt.imshow(self.indoor[:,:,0], cmap=plt.get_cmap('gray'))
        plt.title("indoor red RGB")
        plt.show()

        ##indoor Green grayscale
        plt.imshow(self.indoor[:,:,1], cmap=plt.get_cmap('gray'))
        plt.title("indoor green RGB")
        plt.show()

        ##indoor Blue grayscale
        plt.imshow(self.indoor[:,:,2], cmap=plt.get_cmap('gray'))
        plt.title("indoor blue RGB")
        plt.show()


        ##TEST
        plt.imshow(self.indoor[:,:,3], cmap=plt.get_cmap('gray'))
        plt.title("indoor HIDDEN layer grayscale")
        plt.show()

        #outdoor image
        plt.imshow(self.outdoor, cmap=plt.get_cmap('gray'))
        plt.title("outdoor image")
        plt.show()
    
        ##outdoor Red grayscale
        plt.imshow(self.outdoor[:,:,0], cmap=plt.get_cmap('gray'))
        plt.title("outdoor red RGB")
        plt.show()

        ##outdoor Green grayscale
        plt.imshow(self.outdoor[:,:,1], cmap=plt.get_cmap('gray'))
        plt.title("outdoor green RGB")
        plt.show()

        ##outdoor Blue grayscale
        plt.imshow(self.outdoor[:,:,2], cmap=plt.get_cmap('gray'))
        plt.title("outdoor blue RGB")
        plt.show()

        ##outdoor HIDDEN grayscale
        plt.imshow(self.outdoor[:,:,3], cmap=plt.get_cmap('gray'))
        plt.title("outdoor hidden layer grayscale")
        plt.show()

        indoor_lab = color.rgb2lab(self.indoor[:,:,:3])
        outdoor_lab = color.rgb2lab(self.outdoor[:,:,:3])


        ##LAB indoor L
        plt.imshow(indoor_lab[:,:,0], cmap=plt.get_cmap('gray'))
        plt.title("indoor L LAB")
        plt.show()

        ##LAB indoor A
        plt.imshow(indoor_lab[:,:,1], cmap=plt.get_cmap('gray'))
        plt.title("indoor A LAB")
        plt.show()

        ##LAB indoor B
        plt.imshow(indoor_lab[:,:,2], cmap=plt.get_cmap('gray'))
        plt.title("indoor B LAB")
        plt.show()

        ##LAB outdoor L
        plt.imshow(outdoor_lab[:,:,0], cmap=plt.get_cmap('gray'))
        plt.title("outdoor L LAB")
        plt.show()

        ##LAB outdoor A
        plt.imshow(outdoor_lab[:,:,1], cmap=plt.get_cmap('gray'))
        plt.title("outdoor A LAB")
        plt.show()

        ##LAB outdoor B
        plt.imshow(outdoor_lab[:,:,2], cmap=plt.get_cmap('gray'))
        plt.title("outdoor B LAB")
        plt.show()
        

        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        uncast_img = self.img[:,:,:3]
        img = uncast_img.astype(np.double)
        img = img/255
        img_red = img[:,:, 0]
        img_green = img[:,:,1]
        img_blue = img[:,:,2]

        shapes = img_red.shape

        H = np.zeros(shapes)
        S = np.zeros(shapes)
        V = np.zeros(shapes)



        max = np.amax(img, axis=2)
        min = np.amin(img, axis=2)

        V = max

        for idx, val in np.ndenumerate(img[:,:,0]):
            val = img[idx]
            red = val[0]
            green = val[1]
            blue = val[2]

            max = np.max(val)
            min = np.min(val)

            V[idx] = max
            c = max - min

            if max != 0:
                S[idx] = c / max

            h_prime = None

            if c == 0:
                H[idx] = 0
                continue
            if max == red:
                h_prime = (green - blue) / c
            elif max == green:
                h_prime = (blue - red) / c + 2
            elif max == blue:
                h_prime = (red - green) / c + 4

            if h_prime < 0:
                H[idx] = h_prime / 6 + 1
            else:
                H[idx] = h_prime / 6

        HSV = np.stack([H, S, V], axis=2)
        #print(img.shape)
        #plt.imshow(uncast_img, cmap=plt.get_cmap('gray'))
        #plt.title("outdoor")
        #plt.show()
        #plt.imshow(HSV, cmap=plt.get_cmap('gray'))
        #plt.title("HSV outdoor ")
        #plt.show()
            
        
        #print(np.sum(np.any(img, axis=2, where = [0,0,0])))
        ########debug###########
        #print(img[-2, 1, :])
        #print(np.sum(np.where(V == 0, 1, 0)))
        #print(np.sum(np.where(img[:,:] == [0,0,0], 1, 0)))

        #C = V - min
        #S = C / V


        #h_prime = np.zeros(shape = V.shape)

        #print(h_prime.shape)

        #for idx, val in np.ndenumerate(V):
            #print(idx)
        #    if C[idx] == 0:
        #        h_prime[idx] = 0
        #    elif val == img_red[idx]:
        #        h_prime[idx] = (img_green[idx] - img_blue[idx]) / C[idx]
        #    elif val == img_green[idx]:
        #        h_prime[idx] = (img_blue[idx] - img_red[idx]) / C[idx] + 2
        #    elif val == img_blue[idx]:
        #        h_prime[idx] = (img_red[idx] - img_green[idx]) / C[idx] + 4

        #h_prime = np.where(V == img_blue, (img_red - img_green) / C , None)
        #h_prime = np.where(V == img_green, (img_blue- img_red) / C + 2 , None)
        #h_prime = np.where(V == img_red, (img_green - img_blue) / C + 4 , None)

        #h_prime = np.where(C == 0, None , 
        #    np.where(V == img_red, (img_green - img_blue) / C , 
        #        np.where(V == img_green, (img_blue - img_red) / C + 2 , 
        #            np.where(V == img_blue, (img_red - img_green) / C + 4, None))))

        
        #H = np.where(h_prime < 0, h_prime/6 + 1, h_prime / 6)

        #print(H)
        #HSV = np.stack([H, S, V], axis=2)

        #print(HSV)

        plt.imshow(HSV, cmap=plt.get_cmap('gray'))
        plt.title("HSV")
        plt.show()


        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    #p4.prob_4_1()
    HSV = p4.prob_4_2()





