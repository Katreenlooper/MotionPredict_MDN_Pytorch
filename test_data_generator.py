import sys

from numpy.random import normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GaussianDataGenor:
    def __init__(self, mean, std, nsamples):
        self.mean = mean
        self.std = std
        self.n = nsamples
        self.dim = len(mean)

    def gen_samples(self):
        self.pts = normal(loc=self.mean, scale=self.std, size=(self.n, self.dim))
        return self.pts

    def plot_samples(self):
        fig = plt.figure()
        if self.dim == 1:
            plt.hist(self.pts)
        elif self.dim == 2:
            plt.scatter(self.pts[:,0], self.pts[:,1], marker='.')
            plt.axis('equal')
        elif self.dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.mean[0],  self.mean[1],  self.mean[2], c='r', marker='^', s=100)
            ax.scatter(self.pts[:,0], self.pts[:,1], self.pts[:,2], alpha=0.5)
        else:
            sys.exit('Plot error.')
        plt.show()


if __name__ == '__main__':
    
    mean = [0,1,3]
    std = [1]
    n = 1000 # #samples

    GauGenor = GaussianDataGenor(mean, std, n)
    pts = GauGenor.gen_samples()
    GauGenor.plot_samples()