from Library import Algorithm, Cloud, Manifolds
import numpy as np
import math


def equation(x, t):
    return np.array([math.cos(x[0]) + math.cos(x[1]) - t[0]])



N = 20

n = 2
m = 1

h = 2*6.14/N

xs = [np.linspace(-3.14, 3.14, N), np.linspace(-3.14, 3.14, N)]
x0s = [np.linspace(-3.14, 3.14, 20), np.linspace(-3.14, 3.14, 20)]

t = Cloud.carteisan_product([np.linspace(-2, 2, N)])

barriers, markers = Algorithm.GetBorders(xs, x0s, t, h, equation, 1)

print(barriers)