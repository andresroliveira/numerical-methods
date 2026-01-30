# Numerical Methods

## Introduction

This is a repository containing the implementation of some numerical methods in Python. Methods are lectured in the course `MS211 - Cálculo Numérico` at `Universidade Estadual de Campinas (UNICAMP)`.

The topics of this course are:

- Floating point arithmetic.
- Real function zeros.
- Linear systems.
- Polynomial interpolation.
- Numerical integration.
- Linear least squares.
- Numerical treatment of ordinary differential equations.

This repository is a subset of the topics lectured in the course. Use carefully and always check the results. This is not an official repository of the course.

## Description

This repository contains the implementation of some numerical methods in Python. The methods are divided into the following categories:

### 1. **Optimization** (Root Finding)

Methods for finding zeros of real functions:

- **Bisection** - Bisection method
- **False Position** - Regula Falsi method
- **Newton** - Newton-Raphson method
- **Secant** - Secant method

### 2. **Matrix** (Linear Systems)

Methods for solving linear systems and matrix decomposition:

- **LU Decomposition** - LU factorization
- **QR Decomposition** - QR factorization
- **Jacobi** - Jacobi iterative method
- **Gauss-Seidel** - Gauss-Seidel iterative method

### 3. **Interpolation** (Polynomial Interpolation)

Methods for polynomial and piecewise interpolation:

- **Lagrange** - Lagrange polynomial interpolation
- **Newton Divided Differences** - Newton's divided differences
- **Vandermonde** - Vandermonde matrix method
- **Linear Spline** - Piecewise linear interpolation
- **Cubic Spline** - Natural cubic spline interpolation

### 4. **Least Squares** (Curve Fitting)

Methods for data fitting using least squares:

- **Linear Fit** - Linear regression (y = ax + b)
- **Polynomial Fit** - Polynomial regression of arbitrary degree
- **Fourier Fit** - Fourier series approximation (trigonometric functions)
- **Exponential Fit** - Exponential basis fitting
- **General Fit** - Generic least squares with custom basis functions

### 5. **Integration** (Numerical Integration)

Methods for numerical integration:

- **Riemann Sum** - Riemann sum approximation
- **Trapezoidal Rule** - Trapezoidal rule
- **Simpson's Rule** - Simpson's 1/3 rule

### 6. **ODE** (Ordinary Differential Equations)

Methods for solving initial value problems:

- **Euler** - Euler's method
- **Heun** - Heun's method (improved Euler)
- **Runge-Kutta** - Classical 4th order Runge-Kutta

Each method is implemented as a function in a separate module. The modules are organized into directories based on the category of the method.

## Usage

The methods are implemented in the respective directories. To use the methods, you can import the functions from the respective module.

### Example 1: Root Finding

```python
from optimization import bisection

def f(x):
    return x**2 - 2

root = bisection(f, 0, 2, tol=1e-6)
print(root)  # Approximates sqrt(2)
```

### Example 2: Polynomial Interpolation

```python
from interpolation import lagrange

x_data = [0, 1, 2, 3]
y_data = [1, 3, 2, 5]

y = lagrange(x_data, y_data, 1.5)
print(y)  # Interpolated value at x=1.5
```

### Example 3: Least Squares Fitting

```python
from least_squares import linear_fit

x_data = [1, 2, 3, 4, 5]
y_data = [2.1, 3.9, 6.2, 8.1, 9.8]

a, b = linear_fit(x_data, y_data)
print(f"y = {a:.4f}*x + {b:.4f}")
```

### Example 4: Solving ODEs

```python
from ode import runge_kutta

def f(x):
    return x * (1 - x)

result = runge_kutta(f, x0=0.1, T=5, n=100)
# Returns list of solution values at each time step
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
