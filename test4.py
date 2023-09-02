import numpy as np
import math
from Library import Cloud, Manifolds

def inverse(x, t):

    y = np.zeros((2, 2))

    y[0] = x.copy()
    y[1] = x.copy()


    if(np.isnan(x[0])):

        if(abs(t[0] - math.cos(x[1])) < 1):
            y[0, 0] = math.acos(t[0] - math.cos(x[1]))
            y[1, 0] = - math.acos(t[0] - math.cos(x[1]))

    else:
        if(abs(t[0] - math.cos(x[0])) < 1):
            y[0, 1] = math.acos(t[0] - math.cos(x[0]))
            y[1, 1] = - math.acos(t[0] - math.cos(x[0]))

    return y

N = 100

xs = [np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N)]

t = Cloud.carteisan_product([np.linspace(-2, 2, N)])

slices = Cloud.GetSlicesInverse(xs, t, inverse, 1)
cloud = Cloud.Slices2Pointcloud(slices)

# 3d plot cloud

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=0.1, c='k')
plt.show()