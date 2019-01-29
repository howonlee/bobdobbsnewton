import numpy as np
import numpy.random as npr

def rosenbrock(x, y):
    return ((1. - x) ** 2) + (100. * (y - (x ** 2)) ** 2)

def rosenbrock_grad(x, y):
    z1 = -2. + (2. * x) - (400. * x * y) + (400. * (x ** 3))
    z2 = (200. * y) - (200. * (x ** 2))
    return z1, z2

def rosenbrock_inv_grad(z1, z2):
    x = (z1 + 2.) / (2. - (2. * z2))
    y = (z2 / 200.) + (((z1 ** 2) + (4. * z1) + 4.) / (4. - (8. * z2) + (4. * (z2 ** 2))))
    return x, y

if __name__ == "__main__":
    x, y = npr.randn(), npr.randn()
    print(x, y)
    print(rosenbrock_grad(x, y))
    print(rosenbrock_inv_grad(*rosenbrock_grad(x, y)))
