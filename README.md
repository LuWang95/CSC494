# CSC494 Research Project
## Error Propagation in Numerical Integration for Residual Neural ODE Parameter Estimation

### Student
Lu Wang, Guo Youyou

### Supervisor
Prof. Jonathan Calver

---

# Project Overview

This project investigates how numerical integration error propagates through the training process of Residual Neural ODE models and affects:

- gradient and sensitivity calculations,
- learned parameter estimates,
- trajectory prediction accuracy,
- recovery of underlying physical dynamics.

The project focuses on understanding how solver choice and discretization error influence scientific machine learning systems.

---

# Research Questions

The project aims to study:

1. How numerical integration error affects gradient and sensitivity calculations during Neural ODE training.

2. How numerical errors influence learned parameter estimates.

3. Whether improved trajectory accuracy necessarily corresponds to improved recovery of underlying physical parameters.

4. How these effects vary across numerical integration methods and discretization step sizes.

---

# Model Formulation

The project studies Residual Neural ODE models of the form

```math
\frac{dx}{dt}
=
f_{\mathrm{phys}}(x,\phi)
+
r_\theta(x)
```

where

- $f_{\mathrm{phys}}(x,\phi)$ represents known mechanistic dynamics,

- $\phi$ denotes physical parameters,

- $r_\theta(x)$ is a neural residual correction term parameterized by $\theta$.

Experiments will compare:

- classical parameter estimation methods,
- pure Neural ODE models,
- Residual Neural ODE models.

---

# Benchmark Systems

Initial experiments will focus on low-dimensional dynamical systems, including:

- damped nonlinear oscillators,
- Lotka–Volterra predator–prey systems.

Synthetic trajectory data will be generated using high-accuracy numerical solvers to provide:

- reference trajectories,
- ground truth parameters,
- sensitivity baselines.

---

# Numerical Methods

The project will compare multiple numerical integration schemes, including:

- Forward Euler,
- Runge–Kutta methods,
- adaptive solvers,
- multistep methods.

Experiments will vary:

- discretization step size $begin:math:text$ \\Delta t $end:math:text$,
- solver order,
- observational noise level.

---

# Evaluation Metrics

Performance will be evaluated using:

## Trajectory Prediction Error

Given predicted trajectory $begin:math:text$ \\hat\{x\}\(t\) $end:math:text$ and reference trajectory $begin:math:text$ x\(t\) $end:math:text$,

```math
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^N
\|x_i - \hat{x}_i\|^2
```

## Parameter Estimation Error

For estimated parameters $begin:math:text$ \\hat\{\\theta\} $end:math:text$ and ground truth parameters $begin:math:text$ \\theta\^\\ast $end:math:text$,

```math
\mathrm{Relative\ Parameter\ Error}
=
\frac{
\|\hat{\theta}-\theta^\ast\|
}{
\|\theta^\ast\|
}
```

## Gradient Error

The project will compare gradients computed using different numerical solvers against high-accuracy reference gradients:

```math
E_{\mathrm{grad}}
=
\left\|
\nabla_\theta L_{\mathrm{solver}}
-
\nabla_\theta L_{\mathrm{ref}}
\right\|_2
```

## Sensitivity to Numerical Discretization

The project will study how learned dynamics vary under different choices of:

- step size,
- solver order,
- adaptive tolerances.

---

# Tentative Implementation

The project will be implemented using:

- JAX
- Diffrax

Residual dynamics will be represented using small multilayer perceptrons (MLPs).

Classical parameter estimation methods and pure Neural ODE models will first be implemented as baseline models. Residual Neural ODE models will then be introduced and compared against these baselines.

---

# Repository Structure


# Current Status

- [x] Initial proposal
- [x] Baseline parameter estimation
- [x] Pure Neural ODE implementation
- [x] Residual Neural ODE implementation
- [ ] Numerical error propagation experiments
- [ ] Final analysis and report

---

# References

1. Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D.  
   *Neural Ordinary Differential Equations*, NeurIPS 2018.

2. Calver, J., & Enright, W.  
   *Numerical Methods for Computing Sensitivities for ODEs and DDEs*, Numerical Algorithms, 2017.
