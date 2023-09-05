import numpy as np

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point  # The k-dimensional point
        self.left = left    # Left subtree
        self.right = right  # Right subtree


def build_kd_tree(points, depth=0):
    if len(points) == 0:
        return None

    k = points.shape[1]  # Dimensionality of the points
    axis = depth % k    # Select axis based on depth

    # Sort points along the selected axis and choose median as pivot element
    points = points[points[:, axis].argsort()]
    median = len(points) // 2  # Index of median point

    # Create node and construct subtrees
    return Node(
        point=points[median],
        left=build_kd_tree(points[:median], depth + 1),
        right=build_kd_tree(points[median + 1:], depth + 1)
    )


def search_kd_tree(root, target, depth=0):
    if root is None:
        return None

    k = target.shape[0]
    axis = depth % k

    if np.array_equal(target, root.point):
        return root.point
    elif target[axis] < root.point[axis]:
        return search_kd_tree(root.left, target, depth + 1)
    else:
        return search_kd_tree(root.right, target, depth + 1)


def box_search_kd_tree(root, min_bound, max_bound, depth=0, result=[]):
    if root is None:
        return []

    k = min_bound.shape[0]
    axis = depth % k

    if np.all(min_bound <= root.point) and np.all(max_bound >= root.point):
        result.append(root.point)

    if min_bound[axis] <= root.point[axis]:
        box_search_kd_tree(root.left, min_bound, max_bound, depth + 1, result)
    
    if max_bound[axis] >= root.point[axis]:
        box_search_kd_tree(root.right, min_bound, max_bound, depth + 1, result)

    return result

class KDTree:
    def __init__(self, points):
        self.root = build_kd_tree(points)
    
    def search(self, target):
        return search_kd_tree(self.root, target)
    
    def box_search(self, point, dimensions):

        point = np.array(point)
        dimensions = np.array(dimensions)

        min_bound = point - dimensions/2
        max_bound = point + dimensions/2

        return box_search_kd_tree(self.root, min_bound, max_bound)



# test
"""
import matplotlib.pyplot as plt

# Example usage with NumPy arrays:
if __name__ == "__main__":
    points = np.random.rand(30000, 3)
    kd_tree = KDTree(points)
    
    point = np.array([0.5, 0.5, 0.5])
    dimensions = np.array([0.5, 0.5, 0.5])

    results = kd_tree.box_search(point, dimensions)
    results = np.array(results)

    #plot results 3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o', s = 0.1, alpha = 0.1)
    ax.scatter(results[:,0], results[:,1], results[:,2], c='r', marker='o', s = 0.1)

    plt.show()
"""