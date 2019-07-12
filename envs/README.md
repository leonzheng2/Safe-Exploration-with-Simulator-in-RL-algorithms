# OpenAI Gym environments

These environments are implemented in as OpenAI Gym environments. To use them, one can install them in their `gym` Python package, or directly use them by importing the `Environment` classes.

## Linear Quadratic Regulator

See `gym_lqr/`.

`A \in R^{nxn}`, `B \in R^{nxm}`, `Q \in R^{nxn}`, `R \in R^{mxm}`

Linear system: `x[k+1] = A x[k] + B u[k]` where `x[k] \in R^n`: state, `u[k] \in R^m`: action/control input

Cost function: `c[k+1] = x[k]^T Q x[k] + u[k]^T R u[k]`

Controllable: $`\mathrm{rank}(B, AB, A^2B, ...., A^{n-1}B) = n`$

Q, R: symmetric, i.e., `Q^T = Q`, `R^T = R`

Q is positive semi-definite, R is positive definite

## Swimmer Task

See `gym_swimmer/`.

Code adpated from [Remi Coulom's implementation](https://www.remi-coulom.fr/swimmer.tar.bz2).
