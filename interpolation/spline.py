def linear_spline(x_data, y_data, x):
    """
    Linear spline interpolation (piecewise linear interpolation).

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points (must be sorted).
    y_data : list of float
        The y-coordinates of the data points.
    x : float
        The point at which to evaluate the interpolating spline.

    Returns
    -------
    float
        The value of the interpolating spline at x.

    Raises
    ------
    ValueError
        If x is outside the range of x_data.

    """
    n = len(x_data)

    # Check if x is in the valid range
    if x < x_data[0] or x > x_data[-1]:
        raise ValueError(
            f"x = {x} is outside the interpolation range [{x_data[0]}, {x_data[-1]}]"
        )

    # Find the interval containing x
    for i in range(n - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            # Linear interpolation in the interval [x_data[i], x_data[i+1]]
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + t * (y_data[i + 1] - y_data[i])

    # If we reach here, x == x_data[-1]
    return y_data[-1]


def cubic_spline_natural(x_data, y_data, x):
    """
    Natural cubic spline interpolation.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points (must be sorted).
    y_data : list of float
        The y-coordinates of the data points.
    x : float
        The point at which to evaluate the interpolating spline.

    Returns
    -------
    float
        The value of the interpolating spline at x.

    Raises
    ------
    ValueError
        If x is outside the range of x_data.

    """
    n = len(x_data)

    if x < x_data[0] or x > x_data[-1]:
        raise ValueError(
            f"x = {x} is outside the interpolation range [{x_data[0]}, {x_data[-1]}]"
        )

    # Compute differences
    h = [x_data[i + 1] - x_data[i] for i in range(n - 1)]

    # Build tridiagonal system for second derivatives
    # Natural spline: S''(x_0) = S''(x_n) = 0
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    # Boundary conditions
    A[0][0] = 1.0
    A[n - 1][n - 1] = 1.0

    # Interior equations
    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i] -
                    (y_data[i] - y_data[i - 1]) / h[i - 1])

    # Solve tridiagonal system using Thomas algorithm
    c = solve_tridiagonal(A, b)

    # Find the interval containing x
    for i in range(n - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            t = (x - x_data[i]) / h[i]
            a = y_data[i]
            b_coef = (y_data[i + 1] - y_data[i]) / h[i] - h[i] * (2 * c[i] +
                                                                  c[i + 1]) / 3
            d = (c[i + 1] - c[i]) / (3 * h[i])

            return a + b_coef * (x - x_data[i]) + c[i] * (
                x - x_data[i])**2 + d * (x - x_data[i])**3

    return y_data[-1]


def solve_tridiagonal(A, b):
    """
    Solve a tridiagonal system using the Thomas algorithm.

    Parameters
    ----------
    A : list of list of float
        The tridiagonal matrix.
    b : list of float
        The right-hand side vector.

    Returns
    -------
    list of float
        The solution vector.

    """
    n = len(b)
    c_star = [0.0] * n
    d_star = [0.0] * n
    x = [0.0] * n

    # Forward sweep
    c_star[0] = A[0][1] / A[0][0] if n > 1 else 0
    d_star[0] = b[0] / A[0][0]

    for i in range(1, n - 1):
        denom = A[i][i] - A[i][i - 1] * c_star[i - 1]
        c_star[i] = A[i][i + 1] / denom
        d_star[i] = (b[i] - A[i][i - 1] * d_star[i - 1]) / denom

    d_star[n - 1] = (b[n - 1] - A[n - 1][n - 2] * d_star[n - 2]) / (
        A[n - 1][n - 1] - A[n - 1][n - 2] * c_star[n - 2])

    # Back substitution
    x[n - 1] = d_star[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


def main():
    # Example: interpolate f(x) = sin(x)
    import math

    x_data = [0, 0.5, 1.0, 1.5, 2.0]
    y_data = [math.sin(x) for x in x_data]

    # Test linear spline
    x = math.pi / 4
    y_linear = linear_spline(x_data, y_data, x)
    print(f"Linear spline at x = {x:.6f}: {y_linear:.6f}")
    print(f"Actual value: {math.sin(x):.6f}")

    # Test cubic spline
    y_cubic = cubic_spline_natural(x_data, y_data, x)
    print(f"\nCubic spline at x = {x:.6f}: {y_cubic:.6f}")
    print(f"Actual value: {math.sin(x):.6f}")


if __name__ == "__main__":
    main()
