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


def fourier_fit(t_data, y_data, n):
    """
    Fit a Fourier series to data using least squares.

    The fitted function is of the form:
    f(t) = a0 + a1*sin(t) + b1*cos(t) + a2*sin(2t) + b2*cos(2t) + ... + an*sin(nt) + bn*cos(nt)

    Parameters
    ----------
    t_data : list of float
        The t-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    n : int
        The number of harmonics (maximum frequency).

    Returns
    -------
    dict
        A dictionary with keys 'a0', 'a' (list of sin coefficients), 'b' (list of cos coefficients).

    """
    import math

    m = len(t_data)
    # Number of basis functions: 1 + 2*n (constant + n sines + n cosines)
    num_basis = 1 + 2 * n

    # Construct design matrix A
    A = [[0.0] * num_basis for _ in range(m)]

    for i in range(m):
        t = t_data[i]
        A[i][0] = 1.0  # Constant term

        for k in range(1, n + 1):
            A[i][2 * k - 1] = math.sin(k * t)  # sin(kt)
            A[i][2 * k] = math.cos(k * t)  # cos(kt)

    # Solve normal equations
    A_T = transpose(A)
    ATA = matrix_multiply(A_T, A)
    ATb = matrix_vector_multiply(A_T, y_data)

    coeffs = solve_linear_system(ATA, ATb)

    # Extract coefficients
    a0 = coeffs[0]
    a = [coeffs[2 * k - 1] for k in range(1, n + 1)]
    b = [coeffs[2 * k] for k in range(1, n + 1)]

    return {"a0": a0, "a": a, "b": b}


def evaluate_fourier(coeffs, t):
    """
    Evaluate a Fourier series at point t.

    Parameters
    ----------
    coeffs : dict
        Dictionary with 'a0', 'a', and 'b' coefficients.
    t : float
        The point at which to evaluate the series.

    Returns
    -------
    float
        The value of the Fourier series at t.

    """
    import math

    result = coeffs["a0"]
    n = len(coeffs["a"])

    for k in range(1, n + 1):
        result += coeffs["a"][k - 1] * math.sin(k * t)
        result += coeffs["b"][k - 1] * math.cos(k * t)

    return result


def r_squared_fourier(t_data, y_data, coeffs):
    """
    Compute the coefficient of determination (R-squared) for Fourier fit.

    Parameters
    ----------
    t_data : list of float
        The t-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.
    coeffs : dict
        The Fourier coefficients.

    Returns
    -------
    float
        The R-squared value (between 0 and 1).

    """
    y_mean = sum(y_data) / len(y_data)
    ss_tot = sum((y - y_mean)**2 for y in y_data)
    ss_res = sum((y_data[i] - evaluate_fourier(coeffs, t_data[i]))**2
                 for i in range(len(t_data)))

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def main():
    # Example: fit a periodic function
    import math

    # Generate data from f(t) = 1 + 0.5*sin(t) + 0.3*cos(2t)
    t_data = [i * 0.2 for i in range(31)]  # t from 0 to 6
    y_data = [1 + 0.5 * math.sin(t) + 0.3 * math.cos(2 * t) for t in t_data]

    # Add some noise
    import random

    random.seed(42)
    y_data = [y + random.uniform(-0.1, 0.1) for y in y_data]

    # Fit with n=3 harmonics
    n = 3
    coeffs = fourier_fit(t_data, y_data, n)

    print(f"Fitted Fourier series with n={n} harmonics:")
    print(f"  a0 = {coeffs['a0']:.4f}")
    for k in range(1, n + 1):
        print(f"  a{k} = {coeffs['a'][k-1]:.4f}  (sin({k}t))")
        print(f"  b{k} = {coeffs['b'][k-1]:.4f}  (cos({k}t))")

    # Compute R-squared
    r2 = r_squared_fourier(t_data, y_data, coeffs)
    print(f"\nR-squared: {r2:.6f}")

    # Predict a value
    t_new = math.pi / 2
    y_pred = evaluate_fourier(coeffs, t_new)
    print(f"\nPrediction at t = π/2: y = {y_pred:.4f}")


if __name__ == "__main__":
    main()
