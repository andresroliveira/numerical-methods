def linear_fit(x_data, y_data):
    """
    Fit a linear function y = a*x + b to data using least squares.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.

    Returns
    -------
    a : float
        The slope of the fitted line.
    b : float
        The y-intercept of the fitted line.

    """
    n = len(x_data)

    # Compute sums
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xx = sum(x * x for x in x_data)
    sum_xy = sum(x_data[i] * y_data[i] for i in range(n))

    # Solve the normal equations
    # [n      sum_x ] [b]   [sum_y ]
    # [sum_x  sum_xx] [a] = [sum_xy]

    det = n * sum_xx - sum_x * sum_x

    a = (n * sum_xy - sum_x * sum_y) / det
    b = (sum_xx * sum_y - sum_x * sum_xy) / det

    return a, b


def evaluate(a, b, x):
    """
    Evaluate the linear function y = a*x + b at point x.

    Parameters
    ----------
    a : float
        The slope of the line.
    b : float
        The y-intercept of the line.
    x : float
        The point at which to evaluate the function.

    Returns
    -------
    float
        The value of the function at x.

    """
    return a * x + b


def compute_residuals(x_data, y_data, a, b):
    """
    Compute the residuals between data and fitted line.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    a : float
        The slope of the fitted line.
    b : float
        The y-intercept of the fitted line.

    Returns
    -------
    list of float
        The residuals (observed - predicted).

    """
    return [y_data[i] - evaluate(a, b, x_data[i]) for i in range(len(x_data))]


def r_squared(x_data, y_data, a, b):
    """
    Compute the coefficient of determination (R-squared).

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    a : float
        The slope of the fitted line.
    b : float
        The y-intercept of the fitted line.

    Returns
    -------
    float
        The R-squared value (between 0 and 1).

    """
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean) ** 2 for y in y_data)
    ss_res = sum(
        (y_data[i] - evaluate(a, b, x_data[i])) ** 2 for i in range(len(x_data))
    )

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    # Example: fit a line to noisy data
    x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_data = [2.1, 3.9, 6.2, 8.1, 9.8, 12.3, 13.9, 16.1, 18.2, 19.9]

    a, b = linear_fit(x_data, y_data)
    print(f"Fitted line: y = {a:.4f}*x + {b:.4f}")

    # Compute R-squared
    r2 = r_squared(x_data, y_data, a, b)
    print(f"R-squared: {r2:.6f}")

    # Predict a value
    x_new = 5.5
    y_pred = evaluate(a, b, x_new)
    print(f"\nPrediction at x = {x_new}: y = {y_pred:.4f}")


if __name__ == "__main__":
    main()
