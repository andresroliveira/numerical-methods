"""
Methods for solving higher-order ODEs by converting to first-order systems.

A general nth-order ODE:
    y^(n) = f(t, y, y', y'', ..., y^(n-1))

can be converted to a system of first-order ODEs by defining:
    x0 = y
    x1 = y'
    x2 = y''
    ...
    x(n-1) = y^(n-1)

Then:
    x0' = x1
    x1' = x2
    ...
    x(n-2)' = x(n-1)
    x(n-1)' = f(t, x0, x1, ..., x(n-1))
"""


def euler_system(F, x0, T, n):
    """
    Solve a system of first-order ODEs x' = F(t, x) using Euler's method.

    Parameters
    ----------
    F : function
        A function F(t, x) that returns a list representing the derivatives.
        x is a list of state variables.
    x0 : list of float
        The initial conditions [x0(0), x1(0), ..., xn(0)].
    T : float
        The final time.
    n : int
        The number of steps to take.

    Returns
    -------
    list of list of float
        A list where each element is the state vector at each time step.

    """
    x = x0[:]
    dt = T / n
    t = 0
    result = [x0[:]]

    for _ in range(n):
        dx = F(t, x)
        x = [x[i] + dt * dx[i] for i in range(len(x))]
        t += dt
        result.append(x[:])

    return result


def runge_kutta_system(F, x0, T, n):
    """
    Solve a system of first-order ODEs x' = F(t, x) using RK4 method.

    Parameters
    ----------
    F : function
        A function F(t, x) that returns a list representing the derivatives.
        x is a list of state variables.
    x0 : list of float
        The initial conditions [x0(0), x1(0), ..., xn(0)].
    T : float
        The final time.
    n : int
        The number of steps to take.

    Returns
    -------
    list of list of float
        A list where each element is the state vector at each time step.

    """
    x = x0[:]
    dt = T / n
    t = 0
    result = [x0[:]]

    for _ in range(n):
        k1 = F(t, x)
        k2 = F(t + dt / 2, [x[i] + dt * k1[i] / 2 for i in range(len(x))])
        k3 = F(t + dt / 2, [x[i] + dt * k2[i] / 2 for i in range(len(x))])
        k4 = F(t + dt, [x[i] + dt * k3[i] for i in range(len(x))])

        x = [
            x[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
            for i in range(len(x))
        ]
        t += dt
        result.append(x[:])

    return result


def solve_second_order(f, y0, yp0, T, n, method="rk4"):
    """
    Solve a second-order ODE y'' = f(t, y, y') with initial conditions.

    Parameters
    ----------
    f : function
        The function f(t, y, yp) representing y''.
    y0 : float
        The initial value y(0).
    yp0 : float
        The initial value y'(0).
    T : float
        The final time.
    n : int
        The number of steps to take.
    method : str, optional
        The integration method: 'euler' or 'rk4'. Default is 'rk4'.

    Returns
    -------
    tuple of (list of float, list of float)
        Two lists: (y_values, yp_values) at each time step.

    """

    def F(t, x):
        # x[0] = y, x[1] = y'
        return [x[1], f(t, x[0], x[1])]

    x0 = [y0, yp0]

    if method == "euler":
        result = euler_system(F, x0, T, n)
    else:  # rk4
        result = runge_kutta_system(F, x0, T, n)

    y_values = [x[0] for x in result]
    yp_values = [x[1] for x in result]

    return y_values, yp_values


def main():
    # Example 1: Simple harmonic oscillator y'' = -y
    # Exact solution: y(t) = cos(t), y'(t) = -sin(t)
    import math

    def f(t, y, yp):
        return -y  # y'' = -y

    y0 = 1.0  # y(0) = 1
    yp0 = 0.0  # y'(0) = 0
    T = 2 * math.pi
    n = 100

    y_values, yp_values = solve_second_order(f, y0, yp0, T, n, method="rk4")

    print("Simple harmonic oscillator: y'' = -y, y(0) = 1, y'(0) = 0")
    print(f"Solution at t = 2π:")
    print(f"  y computed:  {y_values[-1]:.6f}")
    print(f"  y exact:     {math.cos(T):.6f}")
    print(f"  y' computed: {yp_values[-1]:.6f}")
    print(f"  y' exact:    {-math.sin(T):.6f}")

    # Example 2: Damped oscillator y'' = -2*y' - y
    def g(t, y, yp):
        return -2 * yp - y

    y0 = 1.0
    yp0 = 0.0
    T = 5.0
    n = 100

    y_values2, yp_values2 = solve_second_order(g, y0, yp0, T, n, method="rk4")

    print(f"\nDamped oscillator: y'' = -2y' - y, y(0) = 1, y'(0) = 0")
    print(f"Solution at t = {T}:")
    print(f"  y  = {y_values2[-1]:.6f}")
    print(f"  y' = {yp_values2[-1]:.6f}")


if __name__ == "__main__":
    main()
