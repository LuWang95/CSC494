# CSC494 Research Project
## Error Propagation in Numerical Integration for Residual Neural ODE Parameter Estimation

### Student
Lu Wang  Guo Youyou

### Supervisor
Prof. Jonathan Calver

---

## Project Overview

This project investigates how numerical integration error propagates through the training process of Residual Neural ODE models and affects:

- gradient and sensitivity calculations,
- learned parameter estimates,
- trajectory prediction accuracy,
- and recovery of underlying physical dynamics.

The project focuses on understanding how solver choice and discretization step size influence scientific machine learning systems.

---

## Research Questions

The project aims to study:

1. How numerical integration error affects gradient and sensitivity calculations during Neural ODE training.

2. How these errors influence learned parameter estimates.

3. Whether improved trajectory accuracy necessarily corresponds to improved recovery of underlying physical parameters.

4. How these effects vary across numerical integration methods and discretization step sizes.

---

## Model Formulation

The project studies Residual Neural ODE models of the form:

dx/dt = f_phys(x, φ) + rθ(x)

where:

- f_phys represents known mechanistic dynamics,
- φ denotes physical parameters,
- rθ(x) is a neural residual correction term.

Experiments will compare:

- classical parameter estimation,
- pure Neural ODE models,
- and Residual Neural ODE models.

---

## Benchmark Systems

Initial experiments will focus on low-dimensional dynamical systems, including:

- damped nonlinear oscillators,
- Lotka–Volterra systems.

Synthetic trajectory data will be generated using high-accuracy numerical solvers to provide reference solutions and ground truth parameters.

---

## Numerical Methods

The project will compare multiple numerical integration schemes, including:

- Forward Euler,
- Runge–Kutta methods,
- adaptive solvers,
- and multistep methods.

Experiments will vary:

- discretization step size,
- solver order,
- observational noise level.

---

## Evaluation Metrics

Performance will be evaluated using:

- trajectory prediction error,
- parameter estimation error,
- gradient error,
- and sensitivity to numerical discretization.

---

## Tentative Implementation

The project will be implemented using:

- JAX
- Diffrax

Residual dynamics will be represented using small multilayer perceptrons (MLPs).

Classical parameter estimation methods and pure Neural ODE models will first be implemented as baseline models. Residual Neural ODE models will then be introduced and compared against these baselines.

---

## Repository Structure


---

## Current Status

- [x] Initial proposal
- [ ] Baseline parameter estimation
- [ ] Pure Neural ODE implementation
- [ ] Residual Neural ODE implementation
- [ ] Numerical error propagation experiments
- [ ] Final analysis and report

---

## References

1. Chen et al., Neural Ordinary Differential Equations, NeurIPS 2018.

2. Calver and Enright, Numerical Methods for Computing Sensitivities for ODEs and DDEs, Numerical Algorithms 2017.
