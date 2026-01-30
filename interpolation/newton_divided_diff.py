def divided_differences(x_data, y_data):
    """
    Compute the divided differences table for Newton interpolation.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.

    Returns
    -------
    list of list of float
        The divided differences table.

    """
    n = len(x_data)
    table = [[0.0] * n for _ in range(n)]

    # First column is y_data
    for i in range(n):
        table[i][0] = y_data[i]

    # Compute divided differences
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] -
                           table[i][j - 1]) / (x_data[i + j] - x_data[i])

    return table


def newton_divided_diff(x_data, y_data, x):
    """
    Newton polynomial interpolation using divided differences.

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
    table = divided_differences(x_data, y_data)
    n = len(x_data)
    result = table[0][0]

    product = 1.0
    for i in range(1, n):
        product *= (x - x_data[i - 1])
        result += table[0][i] * product

    return result


def main():
    # Example: interpolate f(x) = sin(x)
    import math

    x_data = [0, 0.5, 1.0, 1.5, 2.0]
    y_data = [math.sin(x) for x in x_data]

    # Evaluate at x = pi/4
    x = math.pi / 4
    y = newton_divided_diff(x_data, y_data, x)
    print(f"Interpolated value at x = {x:.6f}: {y:.6f}")
    print(f"Actual value: {math.sin(x):.6f}")

    # Display divided differences table
    table = divided_differences(x_data, y_data)
    print("\nDivided differences table:")
    for row in table:
        print([f"{val:.4f}" if val != 0 else "0.0000" for val in row])


if __name__ == "__main__":
    main()
