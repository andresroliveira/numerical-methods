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


def general_fit(x_data, y_data, basis_functions):
    """
    Fit a linear combination of arbitrary basis functions to data using least squares.

    The fitted function is of the form:
    f(x) = c0*φ0(x) + c1*φ1(x) + c2*φ2(x) + ... + cn*φn(x)

    where φi are the basis functions provided.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    basis_functions : list of callable
        A list of basis functions. Each function should take a single float
        argument x and return a float.

    Returns
    -------
    list of float
        The coefficients [c0, c1, c2, ...] for the basis functions.

    """
    m = len(x_data)
    n = len(basis_functions)

    # Construct design matrix A
    A = [[0.0] * n for _ in range(m)]

    for i in range(m):
        x = x_data[i]
        for j in range(n):
            A[i][j] = basis_functions[j](x)

    # Solve normal equations
    A_T = transpose(A)
    ATA = matrix_multiply(A_T, A)
    ATb = matrix_vector_multiply(A_T, y_data)

    coeffs = solve_linear_system(ATA, ATb)

    return coeffs


def evaluate_general(coeffs, basis_functions, x):
    """
    Evaluate a linear combination of basis functions at point x.

    Parameters
    ----------
    coeffs : list of float
        The coefficients for each basis function.
    basis_functions : list of callable
        The basis functions.
    x : float
        The point at which to evaluate the function.

    Returns
    -------
    float
        The value of the fitted function at x.

    """
    result = 0.0
    for i in range(len(coeffs)):
        result += coeffs[i] * basis_functions[i](x)

    return result


def r_squared_general(x_data, y_data, coeffs, basis_functions):
    """
    Compute the coefficient of determination (R-squared) for general fit.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    coeffs : list of float
        The fitted coefficients.
    basis_functions : list of callable
        The basis functions.

    Returns
    -------
    float
        The R-squared value (between 0 and 1).

    """
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum(
        (y_data[i] - evaluate_general(coeffs, basis_functions, x_data[i]))**2
        for i in range(len(x_data)))

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    # Example 1: Fit with polynomial basis
    import math

    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [1.1, 2.9, 8.8, 19.2, 34.1, 53.9]

    # Define basis functions: 1, x, x^2
    basis_poly = [lambda x: 1, lambda x: x, lambda x: x**2]

    coeffs_poly = general_fit(x_data, y_data, basis_poly)
    print("Polynomial fit (1, x, x^2):")
    for i, c in enumerate(coeffs_poly):
        print(f"  c{i} = {c:.4f}")

    r2_poly = r_squared_general(x_data, y_data, coeffs_poly, basis_poly)
    print(f"R-squared: {r2_poly:.6f}\n")

    # Example 2: Fit with trigonometric basis
    t_data = [i * 0.2 for i in range(31)]
    y_data_trig = [
        1 + 0.5 * math.sin(t) + 0.3 * math.cos(2 * t) for t in t_data
    ]

    # Define basis functions: 1, sin(t), cos(t), sin(2t), cos(2t)
    basis_trig = [
        lambda t: 1,
        lambda t: math.sin(t),
        lambda t: math.cos(t),
        lambda t: math.sin(2 * t),
        lambda t: math.cos(2 * t),
    ]

    coeffs_trig = general_fit(t_data, y_data_trig, basis_trig)
    print("Trigonometric fit (1, sin(t), cos(t), sin(2t), cos(2t)):")
    labels = ["1", "sin(t)", "cos(t)", "sin(2t)", "cos(2t)"]
    for i, (c, label) in enumerate(zip(coeffs_trig, labels)):
        print(f"  c{i} ({label:8s}) = {c:.4f}")

    r2_trig = r_squared_general(t_data, y_data_trig, coeffs_trig, basis_trig)
    print(f"R-squared: {r2_trig:.6f}\n")

    # Example 3: Fit with exponential basis
    x_data_exp = [i * 0.1 for i in range(21)]
    y_data_exp = [
        2 + 0.5 * math.exp(x) - 0.1 * math.exp(2 * x) for x in x_data_exp
    ]

    # Define basis functions: 1, exp(x), exp(2x)
    basis_exp = [lambda x: 1, lambda x: math.exp(x), lambda x: math.exp(2 * x)]

    coeffs_exp = general_fit(x_data_exp, y_data_exp, basis_exp)
    print("Exponential fit (1, exp(x), exp(2x)):")
    for i, c in enumerate(coeffs_exp):
        print(f"  c{i} = {c:.4f}")

    r2_exp = r_squared_general(x_data_exp, y_data_exp, coeffs_exp, basis_exp)
    print(f"R-squared: {r2_exp:.6f}")


if __name__ == "__main__":
    main()
