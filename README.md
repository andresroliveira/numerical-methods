# Numerical Methods

## Introduction

This is a repository containing the implementation of numerical methods in Python. The methods cover topics from several courses at `Universidade Estadual de Campinas (UNICAMP)`:

- **MS211 - Cálculo Numérico** (Numerical Calculus)
- **MS512 - Análise Numérica I** (Numerical Analysis I)
- **MS612 - Análise Numérica II** (Numerical Analysis II)
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
- Partial differential equations (heat, wave, Poisson)
- Advanced ODE methods (adaptive stepping, multi-step, stiff problems)
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

#### Basic Linear Systems (MS211)

- **Gauss Elimination** - Gaussian elimination with partial pivoting
- **LU Decomposition** - LU factorization
- **QR Decomposition** - QR factorization
- **Jacobi** - Jacobi iterative method
- **Gauss-Seidel** - Gauss-Seidel iterative method

#### Advanced Methods (MS512/MT402)

- **Cholesky** - Cholesky factorization for symmetric positive definite matrices
- **Householder** - Householder reflections for QR decomposition
- **Givens** - Givens rotations for QR decomposition
- **SOR** - Successive Over-Relaxation method
- **Conjugate Gradient** - Conjugate gradient method for large sparse systems
- **Power Method** - Power iteration for finding dominant eigenvalue
- **QR Eigenvalues** - QR algorithm for computing all eigenvalues
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

#### Basic IVP Methods (MS211)

Methods for solving initial value problems:

- **Euler** - Euler's method
- **Heun** - Heun's method (improved Euler)
- **Runge-Kutta** - Classical 4th order Runge-Kutta
- **Taylor Series** - Taylor series method (2nd and 3rd order)
- **Higher Order** - Solving higher-order ODEs via system reduction

#### Advanced Methods (MS612)

Advanced techniques for ODEs:

- **Runge-Kutta-Fehlberg (RK45)** - Adaptive step size control with error estimation
- **Adams-Bashforth** - Explicit multi-step methods (2nd and 4th order)
- **Adams-Moulton** - Implicit multi-step methods (2nd and 4th order)
- **Backward Euler** - Implicit method for stiff problems
- **BDF2** - 2nd order Backward Differentiation Formula for stiff ODEs

### 7. **PDE** (Partial Differential Equations)

Methods for solving PDEs (MS612):

#### Heat Equation (Parabolic)

- **Explicit (FTCS)** - Forward-Time Central-Space method
- **Implicit (BTCS)** - Backward-Time Central-Space method (unconditionally stable)
- **Crank-Nicolson** - Semi-implicit, 2nd order accurate

#### Poisson/Laplace Equation (Elliptic)

- **Jacobi** - Jacobi iteration for 2D Poisson equation
- **Gauss-Seidel** - Gauss-Seidel iteration (faster convergence)
- **SOR** - Successive Over-Relaxation (optimal convergence)

#### Wave Equation (Hyperbolic)

- **Standard Explicit** - Centered differences with CFL condition
- **Lax-Friedrichs** - Dissipative but stable scheme
- **Upwind** - First-order upwind scheme for advection
- **Lax-Wendroff** - Second-order accurate scheme

### 8. **BVP** (Boundary Value Problems)

Methods for solving boundary value problems (MS211):

- **Finite Differences (Linear)** - Finite difference method for linear BVPs
- **Finite Differences (Nonlinear)** - Finite difference with Newton's method for nonlinear BVPs

### 9. **Nonlinear Systems**

Methods for solving systems of nonlinear equations (MS211):

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

### Example 8: Adaptive ODE Solver (RK45)

```python
from ode import rk45

def f(t, y):
    return -10 * y  # Stiff problem

t_vals, y_vals = rk45(f, 0, 1, 2.0, tol=1e-8)
print(f"Solved with {len(t_vals)} adaptive steps")
```

### Example 9: Heat Equation (PDE)

```python
from pde import heat_crank_nicolson
import math

def u0(x):
    return math.sin(math.pi * x)

bc = {'left': 0, 'right': 0}

x, t, u = heat_crank_nicolson(0.1, 0, 1, 51, 1.0, 0.01, u0, bc)
# Solves ∂u/∂t = 0.1 ∂²u/∂x²
```

### Example 10: Poisson Equation (2D PDE)

```python
from pde import poisson_sor

def f(x, y):
    return -2  # Source term

bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 1}

x, y, u, iters = poisson_sor(0, 1, 21, 0, 1, 21, f, bc, omega=1.8)
print(f"Converged in {iters} iterations")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
