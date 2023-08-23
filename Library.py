from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

import tqdm

from itertools import combinations


class Cloud:
    def equation_factory(x, x0, t, find_ids, equation):
        n = len(x) + len(x0)
        x_ = np.zeros(n)

        fixed_ids = []
        for i in range(n):
            if i not in find_ids:
                fixed_ids.append(i)

        x_[find_ids] = x
        x_[fixed_ids] = x0

        return equation(x_, t)


    def carteisan_product(arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
    

    def GetSlices(xs, x0s, t, equation):

        n = len(xs)
        m = t.shape[1]
        r = n - m

        combs = list(combinations(range(n), r))

        slices = {}

        for t0 in tqdm.tqdm(t):

            slices[tuple(t0)] = []

            for c in combs:
                inx_solve = list(c)
                inx_fixed = list(set(range(n)) - set(c))

                x = Cloud.carteisan_product([xs[i] for i in inx_fixed])
                x0 = Cloud.carteisan_product([x0s[i] for i in inx_solve])

                for i in x:
                    for j in x0:
                        y, _, ier, _ = fsolve(Cloud.equation_factory, j, args = (i, t0, inx_solve, equation), xtol = 1e-5, factor=0.1, maxfev=15, full_output=True)
                        if(ier == 1):

                            #print(infodict["fvec"])

                            x_sol = np.zeros(n)
                            x_sol[inx_fixed] = i
                            x_sol[inx_solve] = y
                            slices[tuple(t0)].append(x_sol)
        return slices
    

    def Slices2Pointcloud(slices):

        pointcloud = np.zeros((0,3))

        for key in slices.keys():


            slice = np.array(slices[key])
            t = np.ones((slice.shape[0],1))*key

            if(len(slice) == 0):
                continue

            slice = np.concatenate((t, slice), axis=1)


            pointcloud = np.concatenate((pointcloud, slice), axis=0)


        return pointcloud
