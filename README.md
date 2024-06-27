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

1. **Optimization**: Bisection, False Position, Fixed Point Iteration, Newton-Raphson, Secant.
2. **Matrix**: LU Decomposition, QR Decompositon, Gauss-Seidel, Jacobi
3. **Integration**: Riemann Sum, Trapezoidal Rule, Simpson's Rule, Romberg Integration
4. **ODE**: Euler, Heun, Runge-Kutta

Each method is implemented as a function in a separate module. The modules are organized into directories based on the category of the method. 

**TBD**: Add Polynomial Interpolation, Linear Least Squares. 


## Usage

The methods are implemented in the respective directories. To use the methods, you can import the functions from the respective module. For example, to use the bisection method, you can import the `bisection` function from the `optimization` module as follows:

```python
from optimization import bisection
```

You can then use the `bisection` function to find the root of a function. For example:

```python
root = bisection(f, a, b, tol)
```

where `f` is the function, `a` and `b` are the initial interval, and `tol` is the tolerance.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


