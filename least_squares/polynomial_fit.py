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


def polynomial_fit(x_data, y_data, degree):
    """
    Fit a polynomial of given degree to data using least squares.

    The polynomial is of the form:
    p(x) = c0 + c1*x + c2*x^2 + ... + c_n*x^n

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    degree : int
        The degree of the polynomial to fit.

    Returns
    -------
    list of float
        The coefficients [c0, c1, c2, ...] of the fitted polynomial.

    """
    n = len(x_data)
    m = degree + 1

    # Construct the Vandermonde-like design matrix A
    A = [[x_data[i]**j for j in range(m)] for i in range(n)]

    # Compute A^T * A and A^T * b
    A_T = transpose(A)
    ATA = matrix_multiply(A_T, A)
    ATb = matrix_vector_multiply(A_T, y_data)

    # Solve the normal equations
    coeffs = solve_linear_system(ATA, ATb)

    return coeffs


def evaluate_polynomial(coeffs, x):
    """
    Evaluate a polynomial at point x using Horner's method.

    Parameters
    ----------
    coeffs : list of float
        The polynomial coefficients [c0, c1, c2, ...].
    x : float
        The point at which to evaluate the polynomial.

    Returns
    -------
    float
        The value of the polynomial at x.

    """
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]

    return result


def r_squared_polynomial(x_data, y_data, coeffs):
    """
    Compute the coefficient of determination (R-squared) for polynomial fit.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    coeffs : list of float
        The polynomial coefficients.

    Returns
    -------
    float
        The R-squared value (between 0 and 1).

    """
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum((y_data[i] - evaluate_polynomial(coeffs, x_data[i]))**2
                 for i in range(len(x_data)))

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    # Example: fit a quadratic to noisy data
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [1.1, 2.9, 8.8, 19.2, 34.1, 53.9]  # Approx. y = x^2 + 2x + 1

    # Fit quadratic (degree 2)
    degree = 2
    coeffs = polynomial_fit(x_data, y_data, degree)

    print(f"Fitted polynomial of degree {degree}:")
    for i, c in enumerate(coeffs):
        print(f"  c{i} = {c:.4f}")

    # Compute R-squared
    r2 = r_squared_polynomial(x_data, y_data, coeffs)
    print(f"\nR-squared: {r2:.6f}")

    # Predict a value
    x_new = 2.5
    y_pred = evaluate_polynomial(coeffs, x_new)
    print(f"\nPrediction at x = {x_new}: y = {y_pred:.4f}")
    print(f"Actual (x^2 + 2x + 1): {x_new**2 + 2*x_new + 1:.4f}")


if __name__ == "__main__":
    main()
