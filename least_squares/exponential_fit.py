def transpose(A):
    """
    Transpose a matrix.

    Parameters
    ----------
    A : list of list of float
        The matrix to transpose.

    Returns
    -------
    list of list of float
        The transposed matrix.

    """
    n = len(A)
    m = len(A[0]) if n > 0 else 0
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.

    Parameters
    ----------
    A : list of list of float
        The first matrix.
    B : list of list of float
        The second matrix.

    Returns
    -------
    list of list of float
        The product of A and B.

    """
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C


def matrix_vector_multiply(A, x):
    """
    Multiply a matrix A by a vector x.

    Parameters
    ----------
    A : list of list of float
        The matrix.
    x : list of float
        The vector.

    Returns
    -------
    list of float
        The product of A and x.

    """
    n = len(A)
    result = [0.0] * n

    for i in range(n):
        for j in range(len(x)):
            result[i] += A[i][j] * x[j]

    return result


def solve_linear_system(A, b):
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
    # Create augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i

        # Swap rows
        M[k], M[max_row] = M[max_row], M[k]

        # Eliminate column
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


def exponential_fit(x_data, y_data, n):
    """
    Fit a linear combination of exponentials to data using least squares.

    The fitted function is of the form:
    f(x) = c0 + c1*exp(x) + c2*exp(2x) + ... + cn*exp(nx)

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    n : int
        The maximum exponent.

    Returns
    -------
    list of float
        The coefficients [c0, c1, c2, ..., cn].

    """
    import math

    m = len(x_data)
    # Number of basis functions: 1 + n (constant + n exponentials)
    num_basis = 1 + n

    # Construct design matrix A
    A = [[0.0] * num_basis for _ in range(m)]

    for i in range(m):
        x = x_data[i]
        A[i][0] = 1.0  # Constant term

        for k in range(1, n + 1):
            A[i][k] = math.exp(k * x)

    # Solve normal equations
    A_T = transpose(A)
    ATA = matrix_multiply(A_T, A)
    ATb = matrix_vector_multiply(A_T, y_data)

    coeffs = solve_linear_system(ATA, ATb)

    return coeffs


def evaluate_exponential(coeffs, x):
    """
    Evaluate a linear combination of exponentials at point x.

    Parameters
    ----------
    coeffs : list of float
        The coefficients [c0, c1, c2, ...].
    x : float
        The point at which to evaluate the function.

    Returns
    -------
    float
        The value of the function at x.

    """
    import math

    result = coeffs[0]  # Constant term
    n = len(coeffs) - 1

    for k in range(1, n + 1):
        result += coeffs[k] * math.exp(k * x)

    return result


def r_squared_exponential(x_data, y_data, coeffs):
    """
    Compute the coefficient of determination (R-squared) for exponential fit.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    coeffs : list of float
        The exponential coefficients.

    Returns
    -------
    float
        The R-squared value (between 0 and 1).

    """
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum((y_data[i] - evaluate_exponential(coeffs, x_data[i]))**2
                 for i in range(len(x_data)))

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    # Example: fit exponentials to data
    import math

    # Generate data from f(x) = 2 + 0.5*exp(x) - 0.1*exp(2x)
    x_data = [i * 0.1 for i in range(21)]  # x from 0 to 2
    y_data = [2 + 0.5 * math.exp(x) - 0.1 * math.exp(2 * x) for x in x_data]

    # Add some noise
    import random

    random.seed(42)
    y_data = [y + random.uniform(-0.05, 0.05) for y in y_data]

    # Fit with n=2 exponentials
    n = 2
    coeffs = exponential_fit(x_data, y_data, n)

    print(f"Fitted exponential series with n={n}:")
    print(f"  c0 = {coeffs[0]:.4f}  (constant)")
    for k in range(1, n + 1):
        print(f"  c{k} = {coeffs[k]:.4f}  (exp({k}x))")

    # Compute R-squared
    r2 = r_squared_exponential(x_data, y_data, coeffs)
    print(f"\nR-squared: {r2:.6f}")

    # Predict a value
    x_new = 1.0
    y_pred = evaluate_exponential(coeffs, x_new)
    y_actual = 2 + 0.5 * math.exp(1) - 0.1 * math.exp(2)
    print(f"\nPrediction at x = {x_new}: y = {y_pred:.4f}")
    print(f"Actual (without noise): {y_actual:.4f}")


if __name__ == "__main__":
    main()
