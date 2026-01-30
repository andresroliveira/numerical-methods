# Numerical Methods

## Introduction

This is a repository containing the implementation of numerical methods in Python. The methods cover topics from several courses at `Universidade Estadual de Campinas (UNICAMP)`:

- **MS211 - Cálculo Numérico** (Numerical Calculus)
- **MS512 - Análise Numérica I** (Numerical Analysis I)
- **MT402 - Matrizes** (Matrices)

### Main Topics Covered

- Floating point arithmetic
- Real function zeros
- Linear systems (direct and iterative methods)
- Matrix factorizations (LU, QR, Cholesky)
- Eigenvalue computations
- Polynomial interpolation
- Numerical integration
- Least squares approximation
- Ordinary differential equations (IVP and BVP)
- Systems of nonlinear equations

All implementations use pure Python (no external libraries like NumPy or SciPy) for educational purposes. Use carefully and always verify results. This is not an official repository of these courses.

## Description

This repository contains the implementation of some numerical methods in Python. The methods are divided into the following categories:

### 1. **Optimization** (Root Finding)

Methods for finding zeros of real functions:

- **Bisection** - Bisection method
- **False Position** - Regula Falsi method
- **Newton** - Newton-Raphson method
- **Secant** - Secant method

### 2. **Matrix** (Linear Systems & Advanced Topics)

#### Basic Methods (MS211)

- **Gauss Elimination** - Gaussian elimination with partial pivoting
- **LU Decomposition** - LU factorization
- **QR Decomposition** - QR factorization
- **Jacobi** - Jacobi iterative method
- **Gauss-Seidel** - Gauss-Seidel iterative method

#### Advanced Methods (MS512/MT402)

- **Cholesky** - Cholesky factorization for symmetric positive definite matrices
- **SOR** - Successive Over-Relaxation method
- **Conjugate Gradient** - Conjugate gradient method for large sparse systems
- **Power Method** - Power iteration for finding dominant eigenvalue
- **Matrix Norms** - Matrix and vector norm computations (1-norm, ∞-norm, Frobenius)

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
- **Gaussian Quadrature** - Gauss-Legendre quadrature (2, 3, and 4 points)

### 6. **ODE** (Ordinary Differential Equations)

Methods for solving initial value problems:

- **Euler** - Euler's method
- **Heun** - Heun's method (improved Euler)
- **Runge-Kutta** - Classical 4th order Runge-Kutta
- **Taylor Series** - Taylor series method (2nd and 3rd order)
- **Higher Order** - Solving higher-order ODEs via system reduction

### 7. **BVP** (Boundary Value Problems)

Methods for solving boundary value problems:

- **Finite Differences (Linear)** - Finite difference method for linear BVPs
- **Finite Differences (Nonlinear)** - Finite difference with Newton's method for nonlinear BVPs

### 8. **Nonlinear Systems**

Methods for solving systems of nonlinear equations:

- **Newton System** - Newton's method for systems of nonlinear equations

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

### Example 5: Gaussian Quadrature

```python
from integration import gaussian_quadrature
import math

def f(x):
    return math.sin(x)

integral = gaussian_quadrature(f, 0, math.pi, n=3)
print(integral)  # Approximates 2.0
```

### Example 6: Boundary Value Problem

```python
from bvp import finite_differences_linear

def p(x):
    return 0

def q(x):
    return 0

def r(x):
    return -2

x, y = finite_differences_linear(p, q, r, 0, 1, 0, 0, n=10)
# Solves y'' = -2, y(0) = 0, y(1) = 0
```

### Example 7: Nonlinear Systems

```python
from nonlinear_systems import newton_system

def F(x):
    return [x[0]**2 + x[1]**2 - 5, x[0]*x[1] - 2]

def J(x):
    return [[2*x[0], 2*x[1]], [x[1], x[0]]]

solution = newton_system(F, J, [1.5, 1.5])
print(solution)  # Solves x^2 + y^2 = 5, xy = 2
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
