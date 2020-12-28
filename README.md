# MadamOpt.jl

[![mit_badge]][mit_url]
[![docs_badge]][docs_url]

## Summary

This is a testing ground for some extensions related to [Adam (Adaptive Moment
Estimation)][adam] using [Julia][julia].  MadamOpt.jl was born out of a need for
gradient-free online optimization.

Note that while this library could be used to train deep models, that is not
its chief design goal. Nevertheless, an example of using MadamOpt with
[FluxML][flux] is included in the `examples` directory (the library supports
GPU acceleration / CUDA when a gradient is provided).

## Features

The extensions currently implemented by the library are:
- L1 regularization via [ISTA (Iterative Shrinkage-Thresholding Algorithm)][ista].
- Gradient-free optimization via a discrete approximation of the gradient using
  a subset of model parameters at each iteration (suitable for small to
  medium-sized models).
- A technique loosely based on [simulated annealing][annealing] for estimating
  non-convex functions without using a gradient.

In the standard Adam, the scaling of the gradient prevents the tresholding from
affecting only relatively insignificant features (i.e. dividing the mean
gradient by square root of the uncentered variance results in a term that
multiplies Adam's alpha term by a value between -1.0 and 1.0, modulo
differences in their decay rates).  Therefore, the step size is further scaled
by `log(1+abs(gradient))`.

See the unit test for examples on fitting a 100-dimensional non-convex Ackley
function, a sparse 500x250 matrix, and the Rosenbrock function.

For an API overview, see the [docs][docs_url], unit tests, and examples.

[julia]: https://julialang.org/
[adam]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
[flux]: https://fluxml.ai/
[ista]: https://en.wikipedia.org/wiki/Proximal_gradient_method
[annealing]: https://en.wikipedia.org/wiki/Simulated_annealing
[mit_badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit_url]: https://github.com/trtsl/recon_mcts/blob/master/LICENSE
[docs_badge]: https://img.shields.io/badge/docs-online-blue.svg
[docs_url]: https://trtsl.github.io/MadamOpt/index.html
