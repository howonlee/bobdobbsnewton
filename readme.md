J. R. "Bob" Dobbs Memorial Fast Finite Difference Newton's Method
===

or, a fun little demo of the J. R. "Bob" Dobbs Memorial Inverse Hessian Multiplication Method.

Of course, you can get a closed-form solution to linear regression, but if you want to you can also do gradient descent. Rosenbrock, you need an optimization thing. Works as well as the "slow" Hessian inversion method, which of course doesn't make a real difference with the Rosenbrock function.

I present a method to do Newton's method on the same order as gradient descent (assuming minibatches of same size as the weights, as there's a matrix inversion involved: otherwise, of the order of the matrix inversion only). No bias term, because I didn't feel like it. You can always simulate by mutating the x matrix.

Both gradient of linear regression and the Rosenbrock function, it turns out, is amenable to easy functional inverse, so you can get multiplication of inverse Hessians with gradient really easily according to the [method I found](https://github.com/howonlee/bobdobbshess). So I did, and as you can see if you run it, the thing works gangbusters. If you don't understand that method you're not going to understand this repo.

You can actually take the limit of the derivative and do the actual derivative, not the finite difference, but I highly suspect that won't be possible for the neural network, so I'm studying finite differences.

If you need to learn what regression is in 10 different ways, I recommend _Elements of Statistical Learning_ although I recognize that's a perverse and stupid recommendation and you should take a statistics 101 class or something.

This is continuing on the work I did in the CSPAM thread and you'd probably be best off asking question or making comments there, although I'll probably end up hanging in the HN thread for a while.

I'm still working on doing it for deep nets. There, functional inversion is not easy and is not fun times, numerically.
