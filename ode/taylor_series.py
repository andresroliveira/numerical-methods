"""
Given x' = f(t, x), find x(t) for t in [0, T] with x(0) = x0.
"""


def taylor_series_order2(f, df_dt, df_dx, x0, T, n):
    """
    Solve x' = f(t, x), x(0) = x0, with 2nd order Taylor series method.

    The method uses the expansion:
    x(t + h) ≈ x(t) + h*f(t, x) + (h^2/2)*f'(t, x)

    where f'(t, x) = ∂f/∂t + ∂f/∂x * f

    Parameters
    ----------
    f : function
        The function f(t, x) such that x' = f(t, x).
    df_dt : function
        The partial derivative ∂f/∂t.
    df_dx : function
        The partial derivative ∂f/∂x.
    x0 : float
        The initial condition x(0) = x0.
    T : float
        The final time.
    n : int
        The number of steps to take.

    Returns
    -------
    list of float
        The solution x(t) at each time t.

    """
    x = x0
    dt = T / n
    t = 0
    result = [x0]

    for _ in range(n):
        # Compute total derivative: df/dt = ∂f/∂t + ∂f/∂x * dx/dt
        # Since dx/dt = f, we have df/dt = ∂f/∂t + ∂f/∂x * f
        f_val = f(t, x)
        df_total = df_dt(t, x) + df_dx(t, x) * f_val

        # Taylor series: x(t+h) = x(t) + h*f + (h^2/2)*df
        x += dt * f_val + (dt**2 / 2) * df_total
        t += dt
        result.append(x)

    return result


def taylor_series_order3(f, df_dt, df_dx, d2f_dt2, d2f_dtdx, d2f_dx2, x0, T, n):
    """
    Solve x' = f(t, x), x(0) = x0, with 3rd order Taylor series method.

    Parameters
    ----------
    f : function
        The function f(t, x) such that x' = f(t, x).
    df_dt : function
        The partial derivative ∂f/∂t.
    df_dx : function
        The partial derivative ∂f/∂x.
    d2f_dt2 : function
        The second partial derivative ∂²f/∂t².
    d2f_dtdx : function
        The mixed partial derivative ∂²f/∂t∂x.
    d2f_dx2 : function
        The second partial derivative ∂²f/∂x².
    x0 : float
        The initial condition x(0) = x0.
    T : float
        The final time.
    n : int
        The number of steps to take.

    Returns
    -------
    list of float
        The solution x(t) at each time t.

    """
    x = x0
    dt = T / n
    t = 0
    result = [x0]

    for _ in range(n):
        f_val = f(t, x)
        df_dt_val = df_dt(t, x)
        df_dx_val = df_dx(t, x)

        # First derivative: df/dt
        df_total = df_dt_val + df_dx_val * f_val

        # Second derivative: d²f/dt²
        d2f_total = (
            d2f_dt2(t, x)
            + 2 * d2f_dtdx(t, x) * f_val
            + d2f_dx2(t, x) * f_val * f_val
            + df_dx_val * df_total
        )

        # Taylor series: x(t+h) = x + h*f + (h^2/2)*df + (h^3/6)*d2f
        x += dt * f_val + (dt**2 / 2) * df_total + (dt**3 / 6) * d2f_total
        t += dt
        result.append(x)

    return result


def main():
    # Example: solve x' = -x, x(0) = 1
    # Exact solution: x(t) = exp(-t)
    import math

    def f(t, x):
        return -x

    def df_dt(t, x):
        return 0  # ∂(-x)/∂t = 0

    def df_dx(t, x):
        return -1  # ∂(-x)/∂x = -1

    x0 = 1.0
    T = 2.0
    n = 10

    result_order2 = taylor_series_order2(f, df_dt, df_dx, x0, T, n)

    print(f"Taylor series (order 2) solution at t={T}:")
    print(f"  Computed: {result_order2[-1]:.6f}")
    print(f"  Exact:    {math.exp(-T):.6f}")
    print(f"  Error:    {abs(result_order2[-1] - math.exp(-T)):.2e}")

    # Example 2: x' = x*(1-x), x(0) = 0.1 (logistic equation)
    def g(t, x):
        return x * (1 - x)

    def dg_dt(t, x):
        return 0

    def dg_dx(t, x):
        return 1 - 2 * x

    x0 = 0.1
    T = 5.0
    n = 50

    result = taylor_series_order2(g, dg_dt, dg_dx, x0, T, n)
    print("\nLogistic equation x' = x(1-x), x(0) = 0.1")
    print(f"Solution at t={T}: {result[-1]:.6f}")


if __name__ == "__main__":
    main()
