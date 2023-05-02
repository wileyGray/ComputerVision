import numpy as np
import matplotlib.pyplot as plt

def prob(n):
    tally = 0
    for i in range(n):
        if np.random.random() <= .2:
            tally += 1
            continue
        elif np.random.random() >= 3/4:
            continue
        if np.random.random() <= 1/3:
            tally += 1
            continue
        elif np.random.random() <= .5:
            tally +=1
            continue
    return tally / n


def prob_1_1(n):
    return np.random.randint(1, 7, size = n)

def prob_1_2():
    y = np.array([11, 22, 33, 44, 55, 66])
    z = np.reshape(y, [3, 2])
    return z

def prob_1_3():
    z = prob_1_2()
    x = np.max(z)
    r, c = np.where(z[:, :] == x)
    c = c[0]
    r = r[0]
    return c, r

def prob_1_4():
    v = np.array([1, 4, 7, 1, 2, 6, 8, 1, 9])
    x = len(np.where(v[:] == 1)[0])
    return x


class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        self.A = np.load("inputAPS1Q2.npy")
        self.A_flat = self.A.flatten()
        self.avg_intensity = np.mean(self.A_flat)
        ###### END CODE HERE ######
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######    
        flat = self.A_flat
        sort = np.sort(flat)[::-1][np.newaxis, :]
        #print(sort)
        

        for index in range(len(sort)):
            sort[0][index] = index 

        #print(sort)
        plt.xlim(1, 0)
        plt.title('intensity diagram')
        plt.xlabel('value')
        plt.imshow(sort, cmap="gray", extent=[0, 1, 0, 1], aspect= 'auto')
        plt.yticks(color = 'w')
        plt.show()

        ###### END CODE HERE ######
        return
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        plt.hist(self.A.flatten(), edgecolor = 'black', bins = 20)
        plt.show()
        ###### END CODE HERE ######
        return
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        X = None
        ###### START CODE HERE ######
        #print(np.arange(400).reshape(4, 10, 10))
        #X = self.A.reshape(4, 50, 50)[0]
        X = self.A[50:, :50]

        #print(X[49,0])
        #print(self.A[99, 0])

        #plt.imshow(self.A)
        #plt.show()
        #plt.imshow(X)
        #plt.show()
        
        ###### END CODE HERE ######
        return X
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        Y = None
        ###### START CODE HERE ######
        Y = self.A_flat
        mean = self.avg_intensity
        for idx, val in np.ndenumerate(Y):
           Y[idx] = val - mean
        
        Y = Y.reshape(100, 100)
        ###### END CODE HERE ######
        return Y
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        Z = None
        ###### START CODE HERE ######
        A = self.A
        t = self.avg_intensity
        shape = (100,100,3)
        Z = np.zeros(shape)
        for idx, val in np.ndenumerate(A):
            if val > t:
                Z[idx][0] = 1
        
        ###### END CODE HERE ######
        return Z


if __name__ == '__main__':
    
    p2 = Prob2()

    
    #p2.prob_2_1()
    #p2.prob_2_2()
    #X = p2.prob_2_3()
    #Y = p2.prob_2_4()
    #Z = p2.prob_2_5()
    print(prob(100000))