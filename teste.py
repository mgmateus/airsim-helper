import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation

from airsim_base.types import Vector3r


def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

#return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])

def plot_bezier_curve(points, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotar pontos de controle
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o')

    # Plotar curva de Bezier
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Curva de Bezier')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def mean_point(position : Vector3r, next_position : Vector3r):
    mean = (position.to_numpy_array() + next_position.to_numpy_array())/2
    mean[0] += mean[0]/10
    mean[1] += mean[1]/10
    mean[2] += mean[2]/10

    triangle = np.array([position.to_numpy_array()])
    triangle = np.vstack((triangle, mean))
    # triangle = np.vstack((triangle, next_position.to_numpy_array()))

    return triangle


# points = np.array([[0,0,0], [0,0,10]])
# path = evaluate_bezier(points, 200)

# p = Vector3r(0, 0, 10)
# t = Vector3r(30, 10, 30)
# points = mean_point(p, t)
# path = np.vstack((path, evaluate_bezier(points, 200)))

# p = Vector3r(30, 10, 30)
# t = Vector3r(30, 80, 30)
# points = mean_point(p, t)
# path = np.vstack((path, evaluate_bezier(points, 200)))

# p = Vector3r(30, 80, 30)
# t = Vector3r(20, 90, 40)
# points = mean_point(p, t)
# path = np.vstack((path, evaluate_bezier(points, 200)))

# plot_bezier_curve(points, path)


def trace(f):
    def wrap(*args, **kwargs):
        print(f"[TRACE] func: {f.__name__}, args: {args}, kwargs: {kwargs}")
        return f(*args, **kwargs) +1

    return wrap

@trace
def teste(a, b):
    return a + b

print(teste(1, 5))


