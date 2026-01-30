"""
Methods for stiff ODEs: Backward Euler and BDF.

Stiff problems have solutions with components that decay at very different rates,
requiring implicit methods for numerical stability.
"""


def backward_euler(f, t0, y0, h, n, tol=1e-8, max_iter=20):
    """
    Solve ODE using Backward Euler method (implicit).

    The method is: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    Solved using fixed-point iteration.

    Parameters
    ----------
    f : callable
        Function f(t, y) that defines dy/dt = f(t, y)
    t0 : float
        Initial time
    y0 : float
        Initial condition
    h : float
        Step size
    n : int
        Number of steps
    tol : float, optional
        Tolerance for fixed-point iteration (default: 1e-8)
    max_iter : int, optional
        Maximum iterations per step (default: 20)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Raises
    ------
    ValueError
        If fixed-point iteration does not converge

    Examples
    --------
    >>> def f(t, y):
    ...     return -10 * y
    >>> t, y = backward_euler(f, 0, 1, 0.1, 10)
    >>> abs(y[-1] - 0.3679) < 0.1  # Stable even with large h
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    for i in range(n):
        t_old = t_vals[-1]
        y_old = y_vals[-1]
        t_new = t_old + h

        # Fixed-point iteration: y_new = y_old + h * f(t_new, y_new)
        # Initial guess: explicit Euler
        y_new = y_old + h * f(t_old, y_old)

        for iteration in range(max_iter):
            y_next = y_old + h * f(t_new, y_new)

            if abs(y_next - y_new) < tol:
                y_new = y_next
                break

            y_new = y_next
        else:
            raise ValueError(
                f"Fixed-point iteration did not converge at t={t_new}")

        t_vals.append(t_new)
        y_vals.append(y_new)

    return t_vals, y_vals


def backward_euler_newton(f, df_dy, t0, y0, h, n, tol=1e-8, max_iter=20):
    """
    Solve ODE using Backward Euler with Newton iteration.

    More robust than fixed-point iteration, requires derivative.

    Parameters
    ----------
    f : callable
        Function f(t, y) that defines dy/dt = f(t, y)
    df_dy : callable
        Partial derivative ∂f/∂y
    t0 : float
        Initial time
    y0 : float
        Initial condition
    h : float
        Step size
    n : int
        Number of steps
    tol : float, optional
        Tolerance for Newton iteration (default: 1e-8)
    max_iter : int, optional
        Maximum iterations per step (default: 20)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -10 * y
    >>> def df_dy(t, y):
    ...     return -10
    >>> t, y = backward_euler_newton(f, df_dy, 0, 1, 0.1, 10)
    >>> import math
    >>> abs(y[-1] - math.exp(-10)) < 0.01
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    for i in range(n):
        t_old = t_vals[-1]
        y_old = y_vals[-1]
        t_new = t_old + h

        # Initial guess: explicit Euler
        y_new = y_old + h * f(t_old, y_old)

        # Newton iteration: solve G(y) = y - y_old - h*f(t_new, y) = 0
        for iteration in range(max_iter):
            G = y_new - y_old - h * f(t_new, y_new)
            dG_dy = 1 - h * df_dy(t_new, y_new)

            if abs(dG_dy) < 1e-14:
                raise ValueError(f"Singular Jacobian at t={t_new}")

            y_next = y_new - G / dG_dy

            if abs(y_next - y_new) < tol:
                y_new = y_next
                break

            y_new = y_next
        else:
            raise ValueError(f"Newton iteration did not converge at t={t_new}")

        t_vals.append(t_new)
        y_vals.append(y_new)

    return t_vals, y_vals


def bdf2(f, t0, y0, h, n, tol=1e-8, max_iter=20):
    """
    Solve ODE using 2nd order Backward Differentiation Formula (BDF2).

    BDF2 formula: y_{n+1} = 4/3*y_n - 1/3*y_{n-1} + 2/3*h*f(t_{n+1}, y_{n+1})

    Parameters
    ----------
    f : callable
        Function f(t, y) that defines dy/dt = f(t, y)
    t0 : float
        Initial time
    y0 : float
        Initial condition
    h : float
        Step size
    n : int
        Number of steps
    tol : float, optional
        Tolerance for iteration (default: 1e-8)
    max_iter : int, optional
        Maximum iterations per step (default: 20)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -10 * y
    >>> t, y = bdf2(f, 0, 1, 0.1, 20)
    >>> import math
    >>> abs(y[-1] - math.exp(-20)) < 0.01
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    # First step: use Backward Euler
    t1 = t0 + h
    y1 = y0 + h * f(t0, y0)  # Initial guess

    for iteration in range(max_iter):
        y_next = y0 + h * f(t1, y1)
        if abs(y_next - y1) < tol:
            y1 = y_next
            break
        y1 = y_next

    t_vals.append(t1)
    y_vals.append(y1)

    # BDF2 for remaining steps
    for i in range(1, n):
        t_old = t_vals[-1]
        y_old = y_vals[-1]
        y_old_minus_1 = y_vals[-2]
        t_new = t_old + h

        # Initial guess: linear extrapolation
        y_new = 2 * y_old - y_old_minus_1

        # Fixed-point iteration
        for iteration in range(max_iter):
            y_next = (4 * y_old - y_old_minus_1 + 2 * h * f(t_new, y_new)) / 3

            if abs(y_next - y_new) < tol:
                y_new = y_next
                break

            y_new = y_next
        else:
            raise ValueError(f"Iteration did not converge at t={t_new}")

        t_vals.append(t_new)
        y_vals.append(y_new)

    return t_vals, y_vals


def main():
    """Example usage of stiff ODE methods."""
    import math

    print("=" * 70)
    print("Methods for Stiff ODEs: Backward Euler and BDF")
    print("=" * 70)

    # Example 1: Moderately stiff problem
    print("\n1. Moderately stiff: dy/dt = -10y, y(0) = 1")
    print("   Exact: y(t) = exp(-10t)")

    def f1(t, y):
        return -10 * y

    def df1_dy(t, y):
        return -10

    h = 0.1
    tf = 1.0
    n = int(tf / h)

    t_be, y_be = backward_euler(f1, 0, 1, h, n)
    t_ben, y_ben = backward_euler_newton(f1, df1_dy, 0, 1, h, n)
    t_bdf2, y_bdf2 = bdf2(f1, 0, 1, h, n)

    y_exact = math.exp(-10)

    print(f"\n   Method\t\t\ty(1.0)\t\t\tError")
    print("   " + "-" * 60)
    print(f"   Exact\t\t\t{y_exact:.10f}")
    print(
        f"   Backward Euler\t\t{y_be[-1]:.10f}\t{abs(y_be[-1] - y_exact):.2e}")
    print(
        f"   Backward Euler (Newton)\t{y_ben[-1]:.10f}\t{abs(y_ben[-1] - y_exact):.2e}"
    )
    print(f"   BDF2\t\t\t{y_bdf2[-1]:.10f}\t{abs(y_bdf2[-1] - y_exact):.2e}")

    # Example 2: Very stiff problem with large step
    print("\n2. Very stiff with large step: dy/dt = -50y, h = 0.25")
    print("   Tests stability with aggressive step size")

    def f2(t, y):
        return -50 * y

    def df2_dy(t, y):
        return -50

    h = 0.25
    tf = 2.0
    n = int(tf / h)

    try:
        t_be, y_be = backward_euler(f2, 0, 1, h, n)
        y_exact = math.exp(-50 * 2)
        error = abs(y_be[-1] - y_exact)
        print(f"\n   Backward Euler: y(2) = {y_be[-1]:.2e}")
        print(f"   Exact:          y(2) = {y_exact:.2e}")
        print(f"   Error:               {error:.2e}")
        print(f"   Method is STABLE even with h*|λ| = {h * 50}")
    except ValueError as e:
        print(f"   Failed: {e}")

    # Example 3: Comparison across methods
    print("\n3. Accuracy comparison: dy/dt = -20y, y(0) = 1, h = 0.05")

    def f3(t, y):
        return -20 * y

    def df3_dy(t, y):
        return -20

    h = 0.05
    tf = 1.0
    n = int(tf / h)

    t_be, y_be = backward_euler(f3, 0, 1, h, n)
    t_ben, y_ben = backward_euler_newton(f3, df3_dy, 0, 1, h, n)
    t_bdf2, y_bdf2 = bdf2(f3, 0, 1, h, n)

    print(f"\n   t\t\tBE\t\tBE-Newton\tBDF2\t\tExact")
    print("   " + "-" * 70)

    for i in [0, len(t_be) // 4, len(t_be) // 2, 3 * len(t_be) // 4, -1]:
        t = t_be[i]
        y_exact = math.exp(-20 * t)
        print(
            f"   {t:.2f}\t\t{y_be[i]:.6f}\t{y_ben[i]:.6f}\t{y_bdf2[i]:.6f}\t{y_exact:.6f}"
        )

    print("\n" + "=" * 70)
    print("Note: Implicit methods are A-stable and handle stiff problems well")
    print("      BDF2 has better accuracy than Backward Euler")
    print("      Newton iteration converges faster than fixed-point")
    print("=" * 70)


if __name__ == "__main__":
    main()
