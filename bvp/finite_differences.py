def solve_tridiagonal(a, b, c, d):
    """
    Solve a tridiagonal system using the Thomas algorithm.

    The system is:
    b[0]*x[0] + c[0]*x[1] = d[0]
    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]  for i=1..n-2
    a[n-1]*x[n-2] + b[n-1]*x[n-1] = d[n-1]

    Parameters
    ----------
    a : list of float
        Lower diagonal (a[0] is not used).
    b : list of float
        Main diagonal.
    c : list of float
        Upper diagonal (c[n-1] is not used).
    d : list of float
        Right-hand side.

    Returns
    -------
    list of float
        The solution vector.

    """
    n = len(b)
    c_star = [0.0] * n
    d_star = [0.0] * n

    # Forward sweep
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i - 1]
        if i < n - 1:
            c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

    # Back substitution
    x = [0.0] * n
    x[n - 1] = d_star[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


def finite_differences_linear(p, q, r, a, b, alpha, beta, n):
    """
    Solve a linear BVP using finite differences.

    The BVP is:
        y'' = p(x)*y' + q(x)*y + r(x)  for x in [a, b]
        y(a) = alpha
        y(b) = beta

    Parameters
    ----------
    p : function
        The coefficient function p(x) for y'.
    q : function
        The coefficient function q(x) for y.
    r : function
        The source term r(x).
    a : float
        The left boundary point.
    b : float
        The right boundary point.
    alpha : float
        The boundary value y(a).
    beta : float
        The boundary value y(b).
    n : int
        The number of interior points (total points = n+2).

    Returns
    -------
    tuple of (list of float, list of float)
        Two lists: (x_values, y_values) representing the solution.

    """
    h = (b - a) / (n + 1)
    x = [a + i * h for i in range(n + 2)]

    # Set up the tridiagonal system
    # For interior points i=1..n:
    # (1/h^2 - p(x_i)/(2h))*y[i-1] + (-2/h^2 + q(x_i))*y[i] + (1/h^2 + p(x_i)/(2h))*y[i+1] = r(x_i)

    lower = [0.0] * n  # a[i]
    diag = [0.0] * n  # b[i]
    upper = [0.0] * n  # c[i]
    rhs = [0.0] * n  # d[i]

    for i in range(n):
        xi = x[i + 1]  # Interior point
        p_val = p(xi)
        q_val = q(xi)
        r_val = r(xi)

        lower[i] = 1 / (h * h) - p_val / (2 * h)
        diag[i] = -2 / (h * h) + q_val
        upper[i] = 1 / (h * h) + p_val / (2 * h)
        rhs[i] = r_val

    # Adjust for boundary conditions
    rhs[0] -= lower[0] * alpha
    rhs[n - 1] -= upper[n - 1] * beta

    # Solve tridiagonal system
    y_interior = solve_tridiagonal(lower, diag, upper, rhs)

    # Combine with boundary values
    y = [alpha] + y_interior + [beta]

    return x, y


def finite_differences_nonlinear(
    f, df_dy, df_dyp, a, b, alpha, beta, n, max_iter=50, tol=1e-6
):
    """
    Solve a nonlinear BVP using finite differences with Newton's method.

    The BVP is:
        y'' = f(x, y, y')  for x in [a, b]
        y(a) = alpha
        y(b) = beta

    Parameters
    ----------
    f : function
        The function f(x, y, yp) representing y''.
    df_dy : function
        The partial derivative ∂f/∂y.
    df_dyp : function
        The partial derivative ∂f/∂y'.
    a : float
        The left boundary point.
    b : float
        The right boundary point.
    alpha : float
        The boundary value y(a).
    beta : float
        The boundary value y(b).
    n : int
        The number of interior points.
    max_iter : int, optional
        Maximum number of Newton iterations. Default is 50.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.

    Returns
    -------
    tuple of (list of float, list of float)
        Two lists: (x_values, y_values) representing the solution.

    """
    h = (b - a) / (n + 1)
    x = [a + i * h for i in range(n + 2)]

    # Initial guess: linear interpolation
    y = [alpha + (beta - alpha) * i / (n + 1) for i in range(n + 2)]

    # Newton iteration
    for iteration in range(max_iter):
        # Build Jacobian and residual for interior points
        J = [[0.0] * n for _ in range(n)]
        residual = [0.0] * n

        for i in range(n):
            xi = x[i + 1]
            yi = y[i + 1]
            yi_prev = y[i]
            yi_next = y[i + 2]

            # Approximate y'
            yp = (yi_next - yi_prev) / (2 * h)

            # Residual: y''_approx - f(x, y, y')
            y_pp_approx = (yi_next - 2 * yi + yi_prev) / (h * h)
            residual[i] = y_pp_approx - f(xi, yi, yp)

            # Jacobian entries
            df_dy_val = df_dy(xi, yi, yp)
            df_dyp_val = df_dyp(xi, yi, yp)

            # d(residual)/d(y[i-1])
            if i > 0:
                J[i][i - 1] = 1 / (h * h) + df_dyp_val / (2 * h)

            # d(residual)/d(y[i])
            J[i][i] = -2 / (h * h) - df_dy_val

            # d(residual)/d(y[i+1])
            if i < n - 1:
                J[i][i + 1] = 1 / (h * h) - df_dyp_val / (2 * h)

        # Adjust for boundaries
        if n > 0:
            residual[0] -= (1 / (h * h) + df_dyp(x[1], y[1], 0) / (2 * h)) * alpha
            residual[n - 1] -= (1 / (h * h) - df_dyp(x[n], y[n], 0) / (2 * h)) * beta

        # Solve J * delta = -residual using Gaussian elimination
        delta = solve_system(J, [-r for r in residual])

        # Update solution
        for i in range(n):
            y[i + 1] += delta[i]

        # Check convergence
        if max(abs(d) for d in delta) < tol:
            break

    return x, y


def solve_system(A, b):
    """
    Solve a linear system Ax = b using Gaussian elimination.

    Parameters
    ----------
    A : list of list of float
        The coefficient matrix.
    b : list of float
        The right-hand side vector.

    Returns
    -------
    list of float
        The solution vector.

    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination
    for k in range(n):
        for i in range(k + 1, n):
            if M[k][k] != 0:
                factor = M[i][k] / M[k][k]
                for j in range(k, n + 1):
                    M[i][j] -= factor * M[k][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if M[i][i] != 0:
            x[i] /= M[i][i]

    return x


def main():
    # Example 1: Linear BVP
    # y'' = -2, y(0) = 0, y(1) = 0
    # Exact solution: y(x) = x(1-x)
    def p(x):
        return 0

    def q(x):
        return 0

    def r(x):
        return -2

    x, y = finite_differences_linear(p, q, r, 0, 1, 0, 0, 10)

    print("Linear BVP: y'' = -2, y(0) = 0, y(1) = 0")
    print("Exact solution: y(x) = x(1-x)")
    print("\nNumerical solution at x = 0.5:")
    mid_idx = len(x) // 2
    print(f"  y(0.5) ≈ {y[mid_idx]:.6f}")
    print(f"  Exact:   {0.5 * 0.5:.6f}")

    # Example 2: Nonlinear BVP
    # y'' = -exp(y), y(0) = 0, y(1) = 0
    def f(x, y, yp):
        import math

        return -math.exp(y)

    def df_dy(x, y, yp):
        import math

        return -math.exp(y)

    def df_dyp(x, y, yp):
        return 0

    x2, y2 = finite_differences_nonlinear(f, df_dy, df_dyp, 0, 1, 0, 0, 10)

    print("\n\nNonlinear BVP: y'' = -exp(y), y(0) = 0, y(1) = 0")
    print(f"Solution at x = 0.5: y ≈ {y2[len(y2) // 2]:.6f}")


if __name__ == "__main__":
    main()
