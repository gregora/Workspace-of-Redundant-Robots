from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

import tqdm

from itertools import combinations


from Library import Cloud


def equation(x, t):
    return np.array([5*math.cos(x[0]) + 3*math.cos(x[1]) - t[0]])

xs = [np.linspace(-3.14, 3.14, 100), np.linspace(-3.14, 3.14, 100)]
x0s = [np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20)]

t = Cloud.carteisan_product([np.linspace(-2, 2, 100)])


slices = Cloud.GetSlices(xs, x0s, t, equation)

cloud = Cloud.Slices2Pointcloud(slices)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=0.1, c='k')

plt.show()



