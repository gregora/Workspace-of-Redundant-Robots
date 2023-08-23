from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

import tqdm

from itertools import combinations


from Library import Cloud, Manifolds

from sklearn.neighbors import KDTree



def equation(x, t):
    return np.array([math.cos(x[0]) + math.cos(x[1]) - t[0]])



N = 20

n = 2
m = 1

h = 2*6.14/N


xs = [np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N)]
x0s = [np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20)]

t = Cloud.carteisan_product([np.linspace(-2, 2, N)])


slices = Cloud.GetSlices(xs, x0s, t, equation, 1)

cloud = Cloud.Slices2Pointcloud(slices)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=0.1, c='k')
plt.show()


ks, trees, manifolds, mappings = Manifolds.ClusterSlices(slices, h)

target_tree = KDTree(t)

barriers = []

for i, point in enumerate(t):
    # find 2*N closest points
    dist, ind = target_tree.query([point], k=2 * m + 1)

    neighbours = ind[0][1:]

    for n in neighbours:
        if (ks[i] > 0) and (ks[n] > 0):

            omega = Manifolds.Match(manifolds[i], manifolds[n], trees[i], trees[n], mappings[i], mappings[n], h = h)
            if(set([]) in omega):
                barriers.append((t[i] + t[n]) / 2)

        elif (ks[i] != 0) or (ks[n] != 0):
            barriers.append((t[i] + t[n]) / 2)

print(barriers)
