# %%
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

# %%
def solve(tx, x):

    # Check if the target is reachable
    d = tx - math.cos(x)
    if d < -1 or d > 1:
        return None # No solution
    
    # Compute the solution
    return math.acos(tx - math.cos(x))

# %%
N = 40

def inverse(tx, visualize = False):

    res_x = np.zeros((2*N, 2))
    res_y = np.zeros((2*N, 2))


    for i, ang in enumerate(np.linspace(-3.14, 3.14, N)):
        y = solve(tx, ang)
        res_x[2*i, 0] = ang
        res_x[2*i, 1] = y

        res_y[2*i, 1] = ang
        res_y[2*i, 0] = y

        res_x[2*i+1, 0] = ang
        res_y[2*i+1, 1] = ang
        if y is not None:
            res_x[2*i+1, 1] = -y
            res_y[2*i+1, 0] = -y
        else:
            res_x[2*i+1, 1] = None
            res_y[2*i+1, 0] = None

    #remove the None values
    #res_x = res_x[~np.isnan(res_x).any(axis=1)]
    #res_y = res_y[~np.isnan(res_y).any(axis=1)]

    if visualize:
        plt.subplot(2, 2, 1)
        plt.scatter(res_x[:, 0], res_x[:, 1], s=1)
        plt.subplot(2, 2, 2)
        plt.scatter(res_y[:, 0], res_y[:, 1], s=1, c='r')
        plt.show()


        plt.scatter(res_x[:, 0], res_x[:, 1], s=1)
        plt.scatter(res_y[:, 0], res_y[:, 1], s=1, c='r')
        plt.show()


    point_cloud = np.concatenate((res_x, res_y))

    return point_cloud



# %%
points = np.zeros((4*N*N, 3))

for i, tx in enumerate(np.linspace(-2, 2, N)):
    point_cloud = inverse(tx)
    points[i*4*N:(i+1)*4*N, 1:] = point_cloud
    points[i*4*N:(i+1)*4*N, 0] = tx

#3d plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


