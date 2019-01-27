import numpy as np
import numpy.random as npr
import numpy.linalg as npl

if __name__ == "__main__":
    npr.seed(1337)
    """
    We have rank issues if we don't have these square
    or at least proper row rank, unfortunately.

    This is a test of the optimization methods, so apologies for
    not having actual data

    Note: if you're going to use this with real data,
    you're going to have to use a regularization.
    The numerics are definitely not happy fun times.
    """
    xs = npr.randn(100, 100)
    ys = npr.randn(100, 100)
    """
    Cache the original unlearned theta so we can see
    how the two algos do on the same minibatch
    """
    original_theta = npr.randn(100, 100)
    print("training with gradient descent...")
    alpha = 1e-3
    theta = original_theta
    for it in range(50000):
        est = np.dot(xs, theta)
        err = 0.5 * np.sum(np.power(est - ys, 2))
        if it % 500 == 0:
            print("it: {}, err: {}".format(str(it), str(err)))
        # derr_dest as in, derivative of error wrt estimate
        derr_dest = est - ys
        derr_dtheta = np.dot(xs.T, derr_dest)
        theta -= alpha * derr_dtheta
    print("resetting theta and redoing with finite difference Newton's...")
    # note we can get away w/ less relaxed relaxation of course
    alpha = 1e-1
    r = 1e-8
    theta = original_theta
    for it in range(500):
        est = np.dot(xs, theta)
        err = 0.5 * np.sum(np.power(est - ys, 2))
        if it % 50 == 0:
            print("it: {}, err: {}".format(str(it), str(err)))
        # derr_dest as in, derivative of error wrt estimate
        derr_dest = est - ys
        derr_dtheta = np.dot(xs.T, derr_dest)
        """
        fd stands for finite difference
        doing one side only fd like a bad person (for demo only)
        v is derr_dtheta itself, so (derr_dtheta + r * derr_dtheta)
        """
        fd_derr_dtheta = derr_dtheta * (1. + r)
        fd_derr_dest = np.dot(npl.inv(xs.T), fd_derr_dtheta)
        fd_est = fd_derr_dest + ys
        # executing the diff here
        fd_theta = (np.dot(npl.inv(xs), fd_est) - theta) / r
        theta -= alpha * fd_theta
