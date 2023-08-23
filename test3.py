from Library import Algorithm, Cloud, Manifolds
import numpy as np
import math

import matplotlib.pyplot as plt


def equation(x, t):
    eq = [0, 0]

    eq[0] = math.cos(x[0]) + math.cos(x[1]) + math.cos(x[2]) - t[0]
    eq[1] = math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2]) - t[1]

    return eq

def equation2(x, t):
    eq = np.zeros(2)

    eq[0] = math.cos(x[0]) + math.cos(x[1])
    eq[1] = math.sin(x[0]) + math.sin(x[1])

    return eq


N = 5

n = 2
m = 1

h = 2*6.14/N

xs = [np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N)]
x0s = [np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20)]

t = Cloud.carteisan_product([np.linspace(-3, 3, N), np.linspace(-3, 3, N)])

print(t.shape)

barriers = Algorithm.GetBorders(xs, x0s, t, h, equation, 2)
barriers = np.array(barriers)

plt.plot(barriers[:, 0], barriers[:, 1])
plt.show()
