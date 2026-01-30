"""
2D Poisson/Laplace Equation: ∇²u = f(x,y)

Laplace equation when f = 0: ∂²u/∂x² + ∂²u/∂y² = 0
Poisson equation when f ≠ 0: ∂²u/∂x² + ∂²u/∂y² = f(x,y)

Implements iterative methods with Dirichlet and Neumann boundary conditions.
"""


def poisson_jacobi(x_min,
                   x_max,
                   nx,
                   y_min,
                   y_max,
                   ny,
                   f,
                   boundary_conditions,
                   tol=1e-6,
                   max_iter=10000):
    """
    Solve 2D Poisson equation using Jacobi iteration.

    Parameters
    ----------
    x_min, x_max : float
        Domain bounds in x
    nx : int
        Number of grid points in x
    y_min, y_max : float
        Domain bounds in y
    ny : int
        Number of grid points in y
    f : callable
        Source term f(x, y)
    boundary_conditions : dict
        {'left': callable/float, 'right': callable/float, 
         'bottom': callable/float, 'top': callable/float}
    tol : float, optional
        Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        Maximum iterations (default: 10000)

    Returns
    -------
    x : list
        x-coordinates
    y : list
        y-coordinates
    u : list of lists
        Solution u[j][i] at (x[i], y[j])
    iterations : int
        Number of iterations performed

    Examples
    --------
    >>> def f(x, y):
    ...     return 0  # Laplace equation
    >>> bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 1}
    >>> x, y, u, iters = poisson_jacobi(0, 1, 11, 0, 1, 11, f, bc, tol=1e-4)
    >>> 0 < u[5][5] < 1  # Interior value between boundaries
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    x = [x_min + i * dx for i in range(nx)]
    y = [y_min + j * dy for j in range(ny)]

    # Initialize solution
    u = [[0.0 for i in range(nx)] for j in range(ny)]
    u_new = [[0.0 for i in range(nx)] for j in range(ny)]

    # Apply boundary conditions
    for i in range(nx):
        bc_bottom = boundary_conditions['bottom']
        bc_top = boundary_conditions['top']
        u[0][i] = bc_bottom(x[i]) if callable(bc_bottom) else bc_bottom
        u[ny - 1][i] = bc_top(x[i]) if callable(bc_top) else bc_top
        u_new[0][i] = u[0][i]
        u_new[ny - 1][i] = u[ny - 1][i]

    for j in range(ny):
        bc_left = boundary_conditions['left']
        bc_right = boundary_conditions['right']
        u[j][0] = bc_left(y[j]) if callable(bc_left) else bc_left
        u[j][nx - 1] = bc_right(y[j]) if callable(bc_right) else bc_right
        u_new[j][0] = u[j][0]
        u_new[j][nx - 1] = u[j][nx - 1]

    # Iteration
    for iteration in range(max_iter):
        max_diff = 0.0

        # Interior points
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # Finite difference for Laplacian
                u_new[j][i] = 0.25 * (u[j][i + 1] + u[j][i - 1] + u[j + 1][i] +
                                      u[j - 1][i] - dx * dx * f(x[i], y[j]))

                diff = abs(u_new[j][i] - u[j][i])
                max_diff = max(max_diff, diff)

        # Update
        for j in range(ny):
            for i in range(nx):
                u[j][i] = u_new[j][i]

        if max_diff < tol:
            return x, y, u, iteration + 1

    raise ValueError(f"Did not converge in {max_iter} iterations")


def poisson_gauss_seidel(x_min,
                         x_max,
                         nx,
                         y_min,
                         y_max,
                         ny,
                         f,
                         boundary_conditions,
                         tol=1e-6,
                         max_iter=10000):
    """
    Solve 2D Poisson equation using Gauss-Seidel iteration.

    Generally converges faster than Jacobi by using updated values immediately.

    Parameters
    ----------
    Same as poisson_jacobi

    Returns
    -------
    Same as poisson_jacobi

    Examples
    --------
    >>> def f(x, y):
    ...     return -2  # ∇²u = -2
    >>> bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
    >>> x, y, u, iters = poisson_gauss_seidel(0, 1, 11, 0, 1, 11, f, bc, tol=1e-4)
    >>> u[5][5] > 0  # Should be positive (bowl shape)
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    x = [x_min + i * dx for i in range(nx)]
    y = [y_min + j * dy for j in range(ny)]

    # Initialize solution
    u = [[0.0 for i in range(nx)] for j in range(ny)]

    # Apply boundary conditions
    for i in range(nx):
        bc_bottom = boundary_conditions['bottom']
        bc_top = boundary_conditions['top']
        u[0][i] = bc_bottom(x[i]) if callable(bc_bottom) else bc_bottom
        u[ny - 1][i] = bc_top(x[i]) if callable(bc_top) else bc_top

    for j in range(ny):
        bc_left = boundary_conditions['left']
        bc_right = boundary_conditions['right']
        u[j][0] = bc_left(y[j]) if callable(bc_left) else bc_left
        u[j][nx - 1] = bc_right(y[j]) if callable(bc_right) else bc_right

    # Iteration
    for iteration in range(max_iter):
        max_diff = 0.0

        # Interior points (use updated values immediately)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_old = u[j][i]

                u[j][i] = 0.25 * (u[j][i + 1] + u[j][i - 1] + u[j + 1][i] +
                                  u[j - 1][i] - dx * dx * f(x[i], y[j]))

                diff = abs(u[j][i] - u_old)
                max_diff = max(max_diff, diff)

        if max_diff < tol:
            return x, y, u, iteration + 1

    raise ValueError(f"Did not converge in {max_iter} iterations")


def poisson_sor(x_min,
                x_max,
                nx,
                y_min,
                y_max,
                ny,
                f,
                boundary_conditions,
                omega=1.5,
                tol=1e-6,
                max_iter=10000):
    """
    Solve 2D Poisson equation using Successive Over-Relaxation (SOR).

    SOR accelerates Gauss-Seidel with relaxation parameter ω.
    Optimal ω ≈ 2/(1 + π/N) for Laplace equation.

    Parameters
    ----------
    omega : float, optional
        Relaxation parameter, 1 < ω < 2 (default: 1.5)
    Other parameters same as poisson_jacobi

    Returns
    -------
    Same as poisson_jacobi

    Examples
    --------
    >>> def f(x, y):
    ...     return 0
    >>> bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': lambda x: x * (1 - x)}
    >>> x, y, u, iters = poisson_sor(0, 1, 21, 0, 1, 21, f, bc, omega=1.8, tol=1e-5)
    >>> iters < 1000  # SOR converges faster
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    x = [x_min + i * dx for i in range(nx)]
    y = [y_min + j * dy for j in range(ny)]

    # Initialize solution
    u = [[0.0 for i in range(nx)] for j in range(ny)]

    # Apply boundary conditions
    for i in range(nx):
        bc_bottom = boundary_conditions['bottom']
        bc_top = boundary_conditions['top']
        u[0][i] = bc_bottom(x[i]) if callable(bc_bottom) else bc_bottom
        u[ny - 1][i] = bc_top(x[i]) if callable(bc_top) else bc_top

    for j in range(ny):
        bc_left = boundary_conditions['left']
        bc_right = boundary_conditions['right']
        u[j][0] = bc_left(y[j]) if callable(bc_left) else bc_left
        u[j][nx - 1] = bc_right(y[j]) if callable(bc_right) else bc_right

    # Iteration
    for iteration in range(max_iter):
        max_diff = 0.0

        # Interior points
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_old = u[j][i]

                # Gauss-Seidel update
                u_gs = 0.25 * (u[j][i + 1] + u[j][i - 1] + u[j + 1][i] +
                               u[j - 1][i] - dx * dx * f(x[i], y[j]))

                # SOR: u_new = ω*u_gs + (1-ω)*u_old
                u[j][i] = omega * u_gs + (1 - omega) * u_old

                diff = abs(u[j][i] - u_old)
                max_diff = max(max_diff, diff)

        if max_diff < tol:
            return x, y, u, iteration + 1

    raise ValueError(f"Did not converge in {max_iter} iterations")


def main():
    """Example usage of 2D Poisson/Laplace solvers."""
    import math

    print("=" * 70)
    print("2D Poisson/Laplace Equation: ∇²u = f(x,y)")
    print("=" * 70)

    # Example 1: Laplace equation with constant boundaries
    print("\n1. Laplace equation (f=0) on unit square")
    print("   BC: u(x,0)=0, u(x,1)=1, u(0,y)=0, u(1,y)=0")

    def f_zero(x, y):
        return 0

    bc1 = {'left': 0, 'right': 0, 'bottom': 0, 'top': 1}

    x, y, u_jac, iters_jac = poisson_jacobi(0,
                                            1,
                                            21,
                                            0,
                                            1,
                                            21,
                                            f_zero,
                                            bc1,
                                            tol=1e-5)
    x, y, u_gs, iters_gs = poisson_gauss_seidel(0,
                                                1,
                                                21,
                                                0,
                                                1,
                                                21,
                                                f_zero,
                                                bc1,
                                                tol=1e-5)
    x, y, u_sor, iters_sor = poisson_sor(0,
                                         1,
                                         21,
                                         0,
                                         1,
                                         21,
                                         f_zero,
                                         bc1,
                                         omega=1.8,
                                         tol=1e-5)

    print(f"\n   Method\t\tIterations\tu(0.5, 0.5)")
    print("   " + "-" * 50)
    print(f"   Jacobi\t\t{iters_jac}\t\t{u_jac[10][10]:.6f}")
    print(f"   Gauss-Seidel\t{iters_gs}\t\t{u_gs[10][10]:.6f}")
    print(f"   SOR (ω=1.8)\t{iters_sor}\t\t{u_sor[10][10]:.6f}")

    # Example 2: Poisson equation with source
    print("\n2. Poisson equation: ∇²u = -2")
    print("   BC: u = 0 on all boundaries")

    def f_const(x, y):
        return -2

    bc2 = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

    x, y, u_poisson, iters = poisson_sor(0,
                                         1,
                                         21,
                                         0,
                                         1,
                                         21,
                                         f_const,
                                         bc2,
                                         omega=1.9,
                                         tol=1e-5)

    print(f"\n   Converged in {iters} iterations")
    print(f"   Maximum u: {max(max(row) for row in u_poisson):.6f} at center")
    print(f"   u(0.5, 0.5) = {u_poisson[10][10]:.6f}")

    # Example 3: Non-constant boundary conditions
    print("\n3. Laplace with varying boundary: u(x,1) = sin(πx)")

    def bc_top(x):
        return math.sin(math.pi * x)

    bc3 = {'left': 0, 'right': 0, 'bottom': 0, 'top': bc_top}

    x, y, u_var, iters = poisson_sor(0,
                                     1,
                                     31,
                                     0,
                                     1,
                                     31,
                                     f_zero,
                                     bc3,
                                     omega=1.85,
                                     tol=1e-6)

    print(f"\n   Converged in {iters} iterations")
    print(f"\n   x\t\ty=0.25\t\ty=0.5\t\ty=0.75\t\ty=1.0")
    print("   " + "-" * 60)

    for i in [0, 7, 15, 23, 30]:
        y_indices = [7, 15, 23, 30]
        values = [u_var[j][i] for j in y_indices]
        print(
            f"   {x[i]:.2f}\t{values[0]:.6f}\t{values[1]:.6f}\t{values[2]:.6f}\t{values[3]:.6f}"
        )

    # Example 4: Convergence comparison
    print("\n4. Convergence rate comparison (same problem)")

    def f_test(x, y):
        return -math.pi**2 * math.sin(math.pi * x) * math.sin(math.pi * y)

    bc4 = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

    _, _, _, iters_jac = poisson_jacobi(0,
                                        1,
                                        21,
                                        0,
                                        1,
                                        21,
                                        f_test,
                                        bc4,
                                        tol=1e-4,
                                        max_iter=50000)
    _, _, _, iters_gs = poisson_gauss_seidel(0,
                                             1,
                                             21,
                                             0,
                                             1,
                                             21,
                                             f_test,
                                             bc4,
                                             tol=1e-4)

    # Find optimal omega
    best_omega = 1.5
    best_iters = 10000
    for omega in [1.5, 1.7, 1.8, 1.85, 1.9, 1.95]:
        try:
            _, _, _, iters = poisson_sor(0,
                                         1,
                                         21,
                                         0,
                                         1,
                                         21,
                                         f_test,
                                         bc4,
                                         omega=omega,
                                         tol=1e-4)
            if iters < best_iters:
                best_iters = iters
                best_omega = omega
        except:
            pass

    print(f"\n   Jacobi:              {iters_jac} iterations")
    print(f"   Gauss-Seidel:        {iters_gs} iterations")
    print(f"   SOR (ω={best_omega}):       {best_iters} iterations")
    print(f"\n   Speedup (SOR/Jacobi): {iters_jac/best_iters:.1f}x")

    print("\n" + "=" * 70)
    print("Notes:")
    print("  - Gauss-Seidel converges ~2x faster than Jacobi")
    print("  - SOR with optimal ω converges much faster than both")
    print("  - Optimal ω depends on grid size and problem")
    print("=" * 70)


if __name__ == "__main__":
    main()
