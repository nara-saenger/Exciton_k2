import numpy as np
import matplotlib.pyplot as plt


d = np.loadtxt('MSD.txt')
plt.plot(d[:,0],d[:,1])
plt.savefig('plot.png')
