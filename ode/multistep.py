"""
Multi-step methods for ODEs: Adams-Bashforth and Adams-Moulton.

These methods use information from multiple previous steps to achieve higher accuracy.
Explicit methods (Adams-Bashforth) are easy to implement but may have stability issues.
Implicit methods (Adams-Moulton) are more stable but require solving equations at each step.
"""


def adams_bashforth_2(f, t0, y0, h, n):
    """
    Solve ODE using 2-step Adams-Bashforth method (explicit).

    Uses RK2 for the first step, then applies AB2 formula.

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

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -y
    >>> t, y = adams_bashforth_2(f, 0, 1, 0.1, 10)
    >>> abs(y[-1] - 0.3679) < 0.01  # y(1) ≈ exp(-1) ≈ 0.3679
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    # Need one additional point - use RK2 (Heun's method)
    t1 = t0 + h
    k1 = f(t0, y0)
    k2 = f(t1, y0 + h * k1)
    y1 = y0 + h * (k1 + k2) / 2

    t_vals.append(t1)
    y_vals.append(y1)

    # Adams-Bashforth 2-step: y_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})
    for i in range(1, n):
        t = t_vals[-1]
        y = y_vals[-1]
        f_n = f(t, y)
        f_n_minus_1 = f(t_vals[-2], y_vals[-2])

        y_next = y + h * (3 * f_n - f_n_minus_1) / 2
        t_next = t + h

        t_vals.append(t_next)
        y_vals.append(y_next)

    return t_vals, y_vals


def adams_bashforth_4(f, t0, y0, h, n):
    """
    Solve ODE using 4-step Adams-Bashforth method (explicit).

    Uses RK4 for the first three steps, then applies AB4 formula.

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

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -2 * t * y
    >>> t, y = adams_bashforth_4(f, 0, 1, 0.1, 20)
    >>> import math
    >>> abs(y[-1] - math.exp(-4)) < 0.001
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    # Need three additional points - use RK4
    for i in range(3):
        t = t_vals[-1]
        y = y_vals[-1]

        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)

        y_next = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_next = t + h

        t_vals.append(t_next)
        y_vals.append(y_next)

    # Adams-Bashforth 4-step
    # y_{n+1} = y_n + h/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
    for i in range(3, n):
        t = t_vals[-1]
        y = y_vals[-1]

        f_n = f(t_vals[-1], y_vals[-1])
        f_n_1 = f(t_vals[-2], y_vals[-2])
        f_n_2 = f(t_vals[-3], y_vals[-3])
        f_n_3 = f(t_vals[-4], y_vals[-4])

        y_next = y + h * (55 * f_n - 59 * f_n_1 + 37 * f_n_2 - 9 * f_n_3) / 24
        t_next = t + h

        t_vals.append(t_next)
        y_vals.append(y_next)

    return t_vals, y_vals


def adams_moulton_2(f, t0, y0, h, n):
    """
    Solve ODE using 2-step Adams-Moulton method (implicit).

    Uses predictor-corrector approach with AB2 as predictor.
    Also known as trapezoidal rule.

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

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -y
    >>> t, y = adams_moulton_2(f, 0, 1, 0.1, 10)
    >>> abs(y[-1] - 0.3679) < 0.001
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    # Need one additional point - use RK2
    t1 = t0 + h
    k1 = f(t0, y0)
    k2 = f(t1, y0 + h * k1)
    y1 = y0 + h * (k1 + k2) / 2

    t_vals.append(t1)
    y_vals.append(y1)

    # Adams-Moulton 2-step (trapezoidal)
    # y_{n+1} = y_n + h/2 * (f_{n+1} + f_n)
    # Use predictor-corrector
    for i in range(1, n):
        t = t_vals[-1]
        y = y_vals[-1]
        t_next = t + h

        # Predictor (AB2)
        f_n = f(t, y)
        f_n_1 = f(t_vals[-2], y_vals[-2])
        y_pred = y + h * (3 * f_n - f_n_1) / 2

        # Corrector (AM2)
        f_next = f(t_next, y_pred)
        y_next = y + h * (f_next + f_n) / 2

        t_vals.append(t_next)
        y_vals.append(y_next)

    return t_vals, y_vals


def adams_moulton_4(f, t0, y0, h, n, max_iter=3):
    """
    Solve ODE using 4-step Adams-Moulton method (implicit).

    Uses predictor-corrector approach with AB4 as predictor.

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
    max_iter : int, optional
        Maximum corrector iterations (default: 3)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Approximate solution values

    Examples
    --------
    >>> def f(t, y):
    ...     return -2 * t * y
    >>> t, y = adams_moulton_4(f, 0, 1, 0.1, 20)
    >>> import math
    >>> abs(y[-1] - math.exp(-4)) < 0.0001
    True
    """
    t_vals = [t0]
    y_vals = [y0]

    # Need three additional points - use RK4
    for i in range(3):
        t = t_vals[-1]
        y = y_vals[-1]

        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)

        y_next = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_next = t + h

        t_vals.append(t_next)
        y_vals.append(y_next)

    # Adams-Moulton 4-step
    # y_{n+1} = y_n + h/24 * (9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
    for i in range(3, n):
        t = t_vals[-1]
        y = y_vals[-1]
        t_next = t + h

        f_n = f(t_vals[-1], y_vals[-1])
        f_n_1 = f(t_vals[-2], y_vals[-2])
        f_n_2 = f(t_vals[-3], y_vals[-3])

        # Predictor (AB4)
        f_n_3 = f(t_vals[-4], y_vals[-4])
        y_pred = y + h * (55 * f_n - 59 * f_n_1 + 37 * f_n_2 - 9 * f_n_3) / 24

        # Corrector (AM4) - iterate for better accuracy
        y_corr = y_pred
        for _ in range(max_iter):
            f_next = f(t_next, y_corr)
            y_corr = y + h * (9 * f_next + 19 * f_n - 5 * f_n_1 + f_n_2) / 24

        t_vals.append(t_next)
        y_vals.append(y_corr)

    return t_vals, y_vals


def main():
    """Example usage of multi-step methods."""
    import math

    print("=" * 70)
    print("Multi-step Methods: Adams-Bashforth and Adams-Moulton")
    print("=" * 70)

    # Example 1: Exponential decay
    print("\n1. Exponential decay: dy/dt = -y, y(0) = 1")
    print("   Exact solution: y(t) = exp(-t)")

    def f1(t, y):
        return -y

    h = 0.1
    tf = 1.0
    n = int(tf / h)

    t_ab2, y_ab2 = adams_bashforth_2(f1, 0, 1, h, n)
    t_ab4, y_ab4 = adams_bashforth_4(f1, 0, 1, h, n)
    t_am2, y_am2 = adams_moulton_2(f1, 0, 1, h, n)
    t_am4, y_am4 = adams_moulton_4(f1, 0, 1, h, n)

    y_exact = math.exp(-1)

    print(f"\n   Method\t\ty(1.0)\t\tError")
    print("   " + "-" * 50)
    print(f"   Exact\t\t{y_exact:.8f}")
    print(f"   AB2\t\t\t{y_ab2[-1]:.8f}\t{abs(y_ab2[-1] - y_exact):.2e}")
    print(f"   AB4\t\t\t{y_ab4[-1]:.8f}\t{abs(y_ab4[-1] - y_exact):.2e}")
    print(f"   AM2\t\t\t{y_am2[-1]:.8f}\t{abs(y_am2[-1] - y_exact):.2e}")
    print(f"   AM4\t\t\t{y_am4[-1]:.8f}\t{abs(y_am4[-1] - y_exact):.2e}")

    # Example 2: Oscillator
    print("\n2. Harmonic oscillator comparison")
    print("   dy/dt = -2ty, y(0) = 1")
    print("   Exact: y(t) = exp(-t²)")

    def f2(t, y):
        return -2 * t * y

    tf = 2.0
    h = 0.1
    n = int(tf / h)

    t_ab4, y_ab4 = adams_bashforth_4(f2, 0, 1, h, n)
    t_am4, y_am4 = adams_moulton_4(f2, 0, 1, h, n)

    print(f"\n   t\t\tAB4\t\tAM4\t\tExact\t\tAB4 Error")
    print("   " + "-" * 70)

    for i in [0, len(t_ab4) // 4, len(t_ab4) // 2, 3 * len(t_ab4) // 4, -1]:
        t = t_ab4[i]
        y_exact = math.exp(-t**2)
        error_ab4 = abs(y_ab4[i] - y_exact)
        print(
            f"   {t:.4f}\t{y_ab4[i]:.8f}\t{y_am4[i]:.8f}\t{y_exact:.8f}\t{error_ab4:.2e}"
        )

    # Example 3: Stability comparison
    print("\n3. Stability test: dy/dt = -10y, y(0) = 1, h = 0.25")
    print("   Large h tests stability of explicit vs implicit methods")

    def f3(t, y):
        return -10 * y

    h = 0.25
    tf = 1.0
    n = int(tf / h)

    try:
        t_ab2, y_ab2 = adams_bashforth_2(f3, 0, 1, h, n)
        print(f"   AB2 (explicit):  y(1) = {y_ab2[-1]:.8f}")
    except:
        print(f"   AB2 (explicit):  UNSTABLE")

    try:
        t_am2, y_am2 = adams_moulton_2(f3, 0, 1, h, n)
        y_exact = math.exp(-10)
        error = abs(y_am2[-1] - y_exact)
        print(
            f"   AM2 (implicit):  y(1) = {y_am2[-1]:.8f} (error: {error:.2e})")
        print(f"   Exact:           y(1) = {y_exact:.8f}")
    except:
        print(f"   AM2 (implicit):  Failed")

    print("\n" + "=" * 70)
    print("Note: Multi-step methods require starting values from RK methods")
    print("      Implicit methods (AM) are more stable than explicit (AB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
