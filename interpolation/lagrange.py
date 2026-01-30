def lagrange(x_data, y_data, x):
    """
    Lagrange polynomial interpolation.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    x : float
        The point at which to evaluate the interpolating polynomial.

    Returns
    -------
    float
        The value of the interpolating polynomial at x.

    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result


def main():
    # Example: interpolate f(x) = sin(x)
    import math

    x_data = [0, 0.5, 1.0, 1.5, 2.0]
    y_data = [math.sin(x) for x in x_data]

    # Evaluate at x = pi/4
    x = math.pi / 4
    y = lagrange(x_data, y_data, x)
    print(f"Interpolated value at x = {x}: {y:.6f}")
    print(f"Actual value: {math.sin(x):.6f}")


if __name__ == "__main__":
    main()
