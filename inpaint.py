import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from skimage import data, io, filters
from math import sqrt, log, acos, pi, floor, exp
from simanneal import Annealer
from random import random, randint
from numpy.linalg import norm

NORM_SCALE = 10
ANGLE_SCALE = 10

def gradient_angle(g1, g2):
    tmp = np.dot(g1.transpose(), g2)
    if norm(g1) * norm(g2) > 0:
        tmp /= norm(g1) * norm(g2)
        # Sometimes the floating point numerical errors causes tmp to be
        # slightly larger than 1.0 or smaller than 1.0
        if tmp > 1.0:
            angle = 0
        elif tmp < -1.0:
            angle = 1.0
        else:
            # Normalize the angle to be between 0 and 1 for convenience
            angle = acos(tmp) / pi
    else:
        angle = 0
    return angle

class GradientInpainter(Annealer):

    """ Use simulated annealer to approximate optimal grayscale levels. """

    def move(self):
        x = randint(x1, x2-1)
        y = randint(y1, y2-1)
        image[x][y] = color[randint(0, len(color)-1)]
        self.state[0][x-x1][y-y1] = image[x][y]

    def energy(self):
        e = 0
        for i in range(x1-2, x2+2):
            for j in range(y1-2, y2+2):
                # Weight the boundary pixels higher so that the hole filling
                # blends well with surroundings
                if i == x1-1 or i == x2 or j == y1-1 or j == y2:
                    w = (y2-y1) * (x2-x1)
                else:
                    w = 1
                gx = (image[i+1][j] - image[i-1][j])/2
                gy = (image[i][j+1] - image[i][j-1])/2
    
                grad[i][j] = np.array([gx, gy])
                ll = hist_gradnorm[int(floor(norm(grad[i][j]) * NORM_SCALE))]
                e += log(ll) * w

        for i in range(x1-1, x2):
            for j in range(y1-1, y2):
                w = 1
                g1 = grad[i][j]
                if i+1 < x2:
                    g2 = grad[i+1][j]
                    angle = gradient_angle(g1, g2)
                    ll = hist_gradangle[int(floor(angle * ANGLE_SCALE))]
                    e += log(ll) * w
                if j+1 < y2:
                    g2 = grad[i][j+1]
                    angle = gradient_angle(g1, g2)
                    ll = hist_gradangle[int(floor(angle * ANGLE_SCALE))]
                    e += log(ll) * w

        return -e

def main():
    if len(sys.argv) != 8:
        print 'Please input filename, hole position, iterations, and Tmax as command line arguments.'
        print 'Example: python inpaint.py circle.png 10 10 20 20 100000 10000'
        exit()

    global image, x1, y1, x2, y2, grad, hist_gradnorm, hist_gradangle, color
        
    filename = str(sys.argv[1])
    image = io.imread(filename, as_grey=True)
    
    x1 = int(sys.argv[2])
    y1 = int(sys.argv[3])
    x2 = int(sys.argv[4])
    y2 = int(sys.argv[5])
    iterations = int(sys.argv[6])
    max_temp = int(sys.argv[7])
    (n1, n2) = image.shape
    
    for i in range(x1, x2):
        for j in range(y1, y2):
            image[i][j] = 0.5

    io.imshow(image)
    io.show()
    
    color = []
    grad = np.empty(image.shape, np.ndarray)
    gradnorm = []
    gradangle = []
    
    indices = np.ndindex(n1, n2)
    indices = filter(lambda (i,j): i not in range(x1, x2) and j not in range(y1,y2), indices)

    for i in range(x1 - (x2-x1), x2 + (x2-x1)):
        for j in range(y1 - (y2-y1), y2 + (y2-y1)):
            if (i,j) in indices:
                color.append(round(image[i][j],2))

    for (i,j) in indices:
        if i in [0, x2]:
            gx = image[i+1][j] - image[i][j]
        elif i in [x1-1, n1-1]:
            gx = image[i][j] - image[i-1][j]
        else:
            gx = (image[i+1][j] - image[i-1][j])/2
    
        if j in [0, y2]:
            gy = image[i][j+1] - image[i][j]
        elif j in [y1-1, n2-1]:
            gy = image[i][j] - image[i][j-1]
        else:
            gy = (image[i][j+1] - image[i][j-1])/2
    
        grad[i][j] = np.array([gx, gy])
    
        gradnorm.append(sqrt(gx**2 + gy**2))
    
    for (i,j) in indices:
        if (i+1,j) in indices:
            g1 = grad[i][j]
            g2 = grad[i+1][j]
            angle = gradient_angle(g1, g2)
            gradangle.append(angle)
        if (i,j+1) in indices:
            g1 = grad[i][j]
            g2 = grad[i][j+1]
            angle = gradient_angle(g1, g2)
            gradangle.append(angle)
            
    bins_gradnorm = np.multiply(np.arange(NORM_SCALE+1), 1.0/NORM_SCALE)
    (hist_gradnorm, bin_edges) = np.histogram(gradnorm, bins=bins_gradnorm, density=True)
    # print hist
    # print bin_edges

    # Modify the 0 entries to slightly positive so that we can take log
    for i in range(hist_gradnorm.size):
        hist_gradnorm[i] /= NORM_SCALE
        if hist_gradnorm[i] == 0:
            hist_gradnorm[i] = 1e-12
    hist_gradnorm = np.append(hist_gradnorm, hist_gradnorm[NORM_SCALE-1])
    
    bins_gradangle = np.multiply(np.arange(ANGLE_SCALE+1), 1.0/ANGLE_SCALE)
    (hist_gradangle, bin_edges) = np.histogram(gradnorm, bins=bins_gradangle, density=True)
    # print hist
    # print bin_edges

    # Modify the 0 entries to slightly positive so that we can take log
    for i in range(hist_gradangle.size):
        hist_gradangle[i] /= ANGLE_SCALE
        if hist_gradangle[i] == 0:
            hist_gradangle[i] = 1e-12
    hist_gradangle = np.append(hist_gradangle, hist_gradangle[ANGLE_SCALE-1])

    # plt.hist(gradnorm, bins=bins_gradnorm)
    # plt.title('Grad norm histogram')
    # plt.show()
    
    # plt.hist(gradangle, bins=bins_gradangle)
    # plt.title('Grad angle histogram')
    # plt.show()

    initial_state = np.empty((x2-x1, y2-y1), float)
    for i in range(x1, x2):
        for j in range(y1, y2):
            image[i][j] = color[randint(0,len(color)-1)]
            initial_state[i-x1][j-y1] = image[i][j]

    annealer = GradientInpainter([initial_state])
    annealer.steps = iterations
    annealer.Tmax = max_temp
    state, energy = annealer.anneal()
    print energy

    for i in range(x1, x2):
        for j in range(y1, y2):
            image[i][j] = state[0][i-x1][j-y1]

    io.imshow(state[0])
    io.show()

    io.imshow(image)
    io.show()

if __name__ == '__main__':
    main()
