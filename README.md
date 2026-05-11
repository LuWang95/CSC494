
# CSC494 Research Project  
## Numerical Error Propagation in Residual Neural ODE Training and Parameter Recovery

### Student
Lu Wang  Youyou Guo

### Supervisor
Prof. Jonathan Calver  

---

# Project Overview

This project investigates how numerical integration error propagates through the training pipeline of Residual Neural ODE models and affects:

- gradient and sensitivity calculations,
- learned parameter estimates,
- trajectory prediction accuracy,
- and recovery of underlying physical dynamics.

The project focuses on understanding how solver choice and discretization step size influence training behavior in scientific machine learning systems.

---

# Research Questions

The project aims to study:

1. How numerical integration error affects gradient and sensitivity calculations during Neural ODE training.

2. How these errors influence learned parameter estimates.

3. Whether improved trajectory accuracy necessarily corresponds to improved recovery of underlying physical parameters.

4. How these effects vary across numerical integration methods and discretization step sizes.

---

# Model Formulation

The project studies Residual Neural ODE models of the form

\[
\frac{dx}{dt}
=
f_{\text{phys}}(x,\phi)
+
r_\theta(x),
\]

where:

- \(f_{\text{phys}}\) represents known mechanistic dynamics,
- \(\phi\) denotes physical parameters,
- \(r_\theta(x)\) is a neural residual correction term.

Experiments will compare:

- classical parameter estimation,
- pure Neural ODE models,
- and Residual Neural ODE models.

---

# Benchmark Systems

Initial experiments will focus on low-dimensional dynamical systems, including:

- damped nonlinear oscillators,
- Lotka--Volterra systems.

Synthetic trajectory data will be generated using high-accuracy numerical solvers.

---

# Numerical Methods

The project will compare multiple numerical integration schemes, including:

- Forward Euler,
- Runge--Kutta methods,
- adaptive solvers,
- and multistep methods.

Experiments will vary:

- discretization step size,
- solver order,
- observational noise level.

---

# Evaluation Metrics

Performance will be evaluated using:

- trajectory prediction error,
- parameter estimation error,
- gradient error,
- and sensitivity to numerical discretization.

---

# Implementation(Tentative)

The project will be implemented using:

- JAX
- Diffrax

Residual dynamics will be represented using small multilayer perceptrons (MLPs).

---

# Repository Structure

