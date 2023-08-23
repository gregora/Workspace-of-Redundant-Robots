from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

import tqdm

from itertools import combinations


from Library import Cloud, Manifolds


def equation(x, t):
    return np.array([math.cos(x[0]) + math.exp(x[1]) - t[0]])



N = 20
h = 2*6.14/N


xs = [np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N)]
x0s = [np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20)]

t = Cloud.carteisan_product([np.linspace(-2, 2, N)])


slices = Cloud.GetSlices(xs, x0s, t, equation)

cloud = Cloud.Slices2Pointcloud(slices)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=0.1, c='k')
plt.show()


ks, trees, manifolds, mappings = Manifolds.ClusterSlices(slices, h)

print(ks)

