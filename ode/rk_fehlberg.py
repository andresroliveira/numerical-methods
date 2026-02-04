"""
Runge-Kutta-Fehlberg (RK45) method with adaptive step size control.

This method uses a pair of Runge-Kutta formulas of orders 4 and 5 to estimate
the local truncation error and automatically adjust the step size.
"""


def rk45(f, t0, y0, tf, tol=1e-6, h_init=0.1, h_min=1e-10, h_max=1.0):
    """
    Solve ODE using Runge-Kutta-Fehlberg method with adaptive step size.

    Uses RK4/RK5 pair for error estimation and step size control.

    Parameters
    ----------
    f : callable
        Function f(t, y) that defines dy/dt = f(t, y)
    t0 : float
        Initial time
    y0 : float or list
        Initial condition
    tf : float
        Final time
    tol : float, optional
        Error tolerance for step size control (default: 1e-6)
    h_init : float, optional
        Initial step size (default: 0.1)
    h_min : float, optional
        Minimum allowed step size (default: 1e-10)
    h_max : float, optional
        Maximum allowed step size (default: 1.0)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list
        Solution values at each time

    Raises
    ------
    ValueError
        If step size becomes too small (indicates stiff problem or bad tolerance)

    Examples
    --------
    Solve dy/dt = -2*t*y, y(0) = 1:

    >>> def f(t, y):
    ...     return -2 * t * y
    >>> t_vals, y_vals = rk45(f, 0, 1, 2, tol=1e-8)
    >>> abs(y_vals[-1] - 0.01832) < 1e-4  # y(2) ≈ exp(-4) ≈ 0.01832
    True
    """
    # Fehlberg coefficients for RK4(5)
    # a coefficients
    a2, a3, a4, a5, a6 = 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2

    # b coefficients
    b21 = 1 / 4
    b31, b32 = 3 / 32, 9 / 32
    b41, b42, b43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
    b51, b52, b53, b54 = 439 / 216, -8, 3680 / 513, -845 / 4104
    b61, b62, b63, b64, b65 = -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40

    # c coefficients for 4th order solution
    c1, c2, c3, c4, c5 = 25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5

    # d coefficients for 5th order solution
    d1, d2, d3, d4, d5, d6 = 16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55

    # Check if y0 is scalar or vector
    is_scalar = isinstance(y0, (int, float))
    if is_scalar:
        y = float(y0)
    else:
        y = list(y0)

    t = t0
    h = h_init
    t_vals = [t]
    y_vals = [y if not is_scalar else y]

    steps = 0
    max_steps = 1000000

    while t < tf:
        if steps > max_steps:
            raise ValueError("Maximum number of steps exceeded")

        # Don't overshoot final time
        if t + h > tf:
            h = tf - t

        # Compute k values
        if is_scalar:
            k1 = h * f(t, y)
            k2 = h * f(t + a2 * h, y + b21 * k1)
            k3 = h * f(t + a3 * h, y + b31 * k1 + b32 * k2)
            k4 = h * f(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
            k5 = h * f(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
            k6 = h * f(
                t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5
            )

            # 4th and 5th order solutions
            y4 = y + c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4 + c5 * k5
            y5 = y + d1 * k1 + d2 * k2 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6

            # Error estimate
            error = abs(y5 - y4)
        else:
            # Vector case
            def add_vectors(v1, *coeffs_vects):
                result = list(v1)
                for coeff, vect in zip(coeffs_vects[::2], coeffs_vects[1::2]):
                    for i in range(len(result)):
                        result[i] += coeff * vect[i]
                return result

            k1 = [h * fi for fi in f(t, y)]
            k2 = [h * fi for fi in f(t + a2 * h, add_vectors(y, b21, k1))]
            k3 = [h * fi for fi in f(t + a3 * h, add_vectors(y, b31, k1, b32, k2))]
            k4 = [
                h * fi
                for fi in f(t + a4 * h, add_vectors(y, b41, k1, b42, k2, b43, k3))
            ]
            k5 = [
                h * fi
                for fi in f(
                    t + a5 * h, add_vectors(y, b51, k1, b52, k2, b53, k3, b54, k4)
                )
            ]
            k6 = [
                h * fi
                for fi in f(
                    t + a6 * h,
                    add_vectors(y, b61, k1, b62, k2, b63, k3, b64, k4, b65, k5),
                )
            ]

            y4 = add_vectors(y, c1, k1, c2, k2, c3, k3, c4, k4, c5, k5)
            y5 = add_vectors(y, d1, k1, d2, k2, d3, k3, d4, k4, d5, k5, d6, k6)

            # Error estimate (max norm)
            error = max(abs(y5[i] - y4[i]) for i in range(len(y4)))

        # Adjust step size
        if error < tol or error == 0:
            # Accept step
            t += h
            y = y5
            t_vals.append(t)
            y_vals.append(y if not is_scalar else y)
            steps += 1

        # Calculate new step size
        if error > 0:
            s = 0.84 * (tol / error) ** 0.25
            h = h * min(4, max(0.1, s))
        else:
            h = h * 4

        # Enforce step size bounds
        h = max(h_min, min(h, h_max))

        if h < h_min and t < tf:
            raise ValueError(
                f"Step size {h} fell below minimum {h_min}. Problem may be stiff or tolerance too tight."
            )

    return t_vals, y_vals


def rkf45_system(f, t0, y0, tf, tol=1e-6, h_init=0.1):
    """
    Convenience wrapper for systems of ODEs.

    Parameters
    ----------
    f : callable
        Function f(t, y) that returns list of derivatives
    t0 : float
        Initial time
    y0 : list
        Initial conditions
    tf : float
        Final time
    tol : float, optional
        Error tolerance (default: 1e-6)
    h_init : float, optional
        Initial step size (default: 0.1)

    Returns
    -------
    t_vals : list
        Time values
    y_vals : list of lists
        Solution values at each time

    Examples
    --------
    Solve system: dy1/dt = y2, dy2/dt = -y1 (harmonic oscillator):

    >>> def f(t, y):
    ...     return [y[1], -y[0]]
    >>> t_vals, y_vals = rkf45_system(f, 0, [1, 0], 6.28, tol=1e-8)
    >>> abs(y_vals[-1][0] - 1) < 0.01  # y1(2π) ≈ 1
    True
    """
    return rk45(f, t0, y0, tf, tol, h_init)


def main():
    """Example usage of RK-Fehlberg method."""
    import math

    print("=" * 60)
    print("Runge-Kutta-Fehlberg (RK45) with Adaptive Step Size")
    print("=" * 60)

    # Example 1: Exponential decay
    print("\n1. Exponential decay: dy/dt = -2ty, y(0) = 1")
    print("   Exact solution: y(t) = exp(-t²)")

    def f1(t, y):
        return -2 * t * y

    t_vals, y_vals = rk45(f1, 0, 1, 2, tol=1e-8)

    print(f"\n   t\t\tRK45\t\tExact\t\tError\t\tSteps: {len(t_vals)}")
    print("   " + "-" * 55)

    for i in [0, len(t_vals) // 4, len(t_vals) // 2, 3 * len(t_vals) // 4, -1]:
        t = t_vals[i]
        y_rk = y_vals[i]
        y_exact = math.exp(-(t**2))
        error = abs(y_rk - y_exact)
        print(f"   {t:.4f}\t{y_rk:.8f}\t{y_exact:.8f}\t{error:.2e}")

    # Example 2: Oscillator (system)
    print("\n2. Harmonic oscillator: y'' + y = 0")
    print("   System: dy1/dt = y2, dy2/dt = -y1")
    print("   Initial: y1(0) = 1, y2(0) = 0")
    print("   Exact: y1(t) = cos(t), y2(t) = -sin(t)")

    def f2(t, y):
        return [y[1], -y[0]]

    t_vals, y_vals = rkf45_system(f2, 0, [1, 0], 2 * math.pi, tol=1e-8)

    print(f"\n   t\t\ty1 (RK45)\ty1 (Exact)\tError\t\tSteps: {len(t_vals)}")
    print("   " + "-" * 60)

    for i in [0, len(t_vals) // 4, len(t_vals) // 2, 3 * len(t_vals) // 4, -1]:
        t = t_vals[i]
        y1_rk = y_vals[i][0]
        y1_exact = math.cos(t)
        error = abs(y1_rk - y1_exact)
        print(f"   {t:.4f}\t{y1_rk:.8f}\t{y1_exact:.8f}\t{error:.2e}")

    # Example 3: Stiff problem (demonstrates adaptive stepping)
    print("\n3. Van der Pol equation (μ = 10): y'' - μ(1-y²)y' + y = 0")
    print("   Demonstrates adaptive step size for stiff-like behavior")

    mu = 10

    def f3(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    try:
        t_vals, y_vals = rkf45_system(f3, 0, [2, 0], 20, tol=1e-5, h_init=0.01)
        print(f"   Successfully solved with {len(t_vals)} adaptive steps")
        print(f"   Final state: y1({t_vals[-1]:.2f}) = {y_vals[-1][0]:.6f}")
    except ValueError as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Note: Adaptive stepping adjusts h to maintain error tolerance")
    print("=" * 60)


if __name__ == "__main__":
    main()
