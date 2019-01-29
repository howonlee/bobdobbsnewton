import numpy as np
import numpy.random as npr

def rosenbrock(x, y):
    return ((1. - x) ** 2) + (100. * (y - (x ** 2)) ** 2)

def rosenbrock_grad(x, y):
    z1 = -2. + (2. * x) - (400. * x * y) + (400. * (x ** 3))
    z2 = (200. * y) - (200. * (x ** 2))
    return z1, z2

def rosenbrock_inv_grad(z1, z2):
    """
    Gotten by a paper and pencil, unfortunately
    """
    x = (z1 + 2.) / (2. - (2. * z2))
    y = (z2 / 200.) + (((z1 ** 2) + (4. * z1) + 4.) / (4. - (8. * z2) + (4. * (z2 ** 2))))
    return x, y

if __name__ == "__main__":
    orig_x, orig_y = npr.randn(), npr.randn()
    print("original x and y: ({}, {})".format(orig_x, orig_y))
    iters = 20
    curr_x, curr_y = orig_x, orig_y
    print("lame gradient method, bounces around a lot")
    alpha = 1e-3
    for it in range(iters):
        val = rosenbrock(curr_x, curr_y)
        print("iter: {}, current val of rosenbrock: {}".format(str(it), str(val)))
        z1, z2 = rosenbrock_grad(curr_x, curr_y)
        curr_x -= alpha * z1
        curr_y -= alpha * z2
    r = 1e-8
    print("final (x,y) with gradient descent: ({}, {})".format(curr_x, curr_y))
    print("finite difference newton's method: look ma, no explicit Hessian!")
    print("resetting x and y back to original values...")
    curr_x, curr_y = orig_x, orig_y
    for it in range(iters):
        val = rosenbrock(curr_x, curr_y)
        print("iter: {}, current val of rosenbrock: {}".format(str(it), str(val)))
        z1, z2 = rosenbrock_grad(curr_x, curr_y)
        fd_x, fd_y = rosenbrock_inv_grad(z1 * (1. + r), z2 * (1. + r))
        curr_x -= (fd_x - curr_x) / r
        curr_y -= (fd_y - curr_y) / r
    print("final (x,y) with newton's method: ({}, {})".format(curr_x, curr_y))
