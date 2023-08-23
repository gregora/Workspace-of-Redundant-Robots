from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import math

import tqdm

from itertools import combinations
from sklearn.neighbors import KDTree


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




class Manifolds:

    def ClusterSlices(slices, h = 0.1):

        ks = []
        trees = []
        manifolds = []
        mappings = []


        for key in slices.keys():
            slice = slices[key]
            slice = np.array(slice)
            
            if(len(slice) == 0):
                ks.append(0)
                trees.append(None)
                manifolds.append([])
                mappings.append({})
                continue

            k, tree, manifolds, mapping = Manifolds.Cluster(slice, h=h)
            
            ks.append(k)
            trees.append(tree)
            manifolds.append(manifolds)
            mappings.append(mapping)

        return ks, trees, manifolds, mappings


    def Cluster(cloud, h = 0.1):

        cloud = cloud.copy()
        cloud = cloud[~np.isnan(cloud).any(axis=1)]

        if(cloud.shape[0] < 1):
            #return 0 and empty tree
            return 0, None, [], {}

        tree = KDTree(cloud)

        k = 0

        manifolds = []
        mapping = {}

        while len(cloud) > 0:
            cloud, manifold = Manifolds.Classify(cloud, tree, cloud[0], h)
            manifolds.append(manifold)

            for point in manifold:
                mapping[point] = k

            k += 1

        return k, tree, manifolds, mapping

    def Classify(cloud, tree, point, h = 0.1):
        #classify all the points in the manifold of which point is a part

        queue = [point]
        explored = set([tuple(point)])

        while len(queue) > 0:

            point = queue.pop(0)
            cloud = cloud[~np.all(cloud == point, axis=1)]

            #find all the points in the neighborhood
            tree_res = tree.query_radius(point.reshape(1, -1), r = h, sort_results = True, return_distance = True) 

            indices = tree_res[0][0]
            indices = indices.astype(int)
            indices = indices[1:]

            for i in indices:
                p = np.asarray(tree.data[i])

                #check that p has not been explored yet and is not in queue
                if not tuple(p) in explored:
                    queue.append(p)
                    explored.add(tuple(p))

        return cloud, explored






    def Match(manifolds_a, manifolds_b, tree_a, tree_b, mapping_a, mapping_b, h = 0.1):
        """
        Match two manifolds using the tree structure
        """

        omega = []

        for m in manifolds_a:
            manifold = list(m)
            omega_m = set([])
            
            for p in manifold:
                neighbours = tree_b.query_radius([p], r = h)[0]

                for n in neighbours:
                    manifold_b = mapping_b[tuple(tree_b.data[n])]
                    omega_m.add(manifold_b)

            omega.append(omega_m)
        return omega

    def DetectManifoldChange(omega1, omega2):
        # Check if there is a change in the manifolds between two points

        if (set([]) in omega1) or (set([]) in omega2):
            return True
        
        return False

