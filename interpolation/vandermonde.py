def vandermonde_matrix(x_data):
    """
    Construct the Vandermonde matrix for interpolation.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.

    Returns
    -------
    list of list of float
        The Vandermonde matrix.

    """
    n = len(x_data)
    V = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            V[i][j] = x_data[i]**j

    return V


def solve_linear_system(A, b):
    """
    Solve a linear system Ax = b using Gaussian elimination with partial pivoting.

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


def vandermonde(x_data, y_data, x):
    """
    Polynomial interpolation using Vandermonde matrix.

    This method solves the linear system V*c = y, where V is the Vandermonde
    matrix and c are the polynomial coefficients. The polynomial is then
    evaluated at the given point x.

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
    # Construct Vandermonde matrix
    V = vandermonde_matrix(x_data)

    # Solve for polynomial coefficients
    coeffs = solve_linear_system(V, y_data)

    # Evaluate polynomial at x
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x**i)

    return result


def get_polynomial_coefficients(x_data, y_data):
    """
    Get the coefficients of the interpolating polynomial.

    Parameters
    ----------
    x_data : list of float
        The x-coordinates of the data points.
    y_data : list of float
        The y-coordinates of the data points.

    Returns
    -------
    list of float
        The polynomial coefficients [c0, c1, c2, ...] where
        p(x) = c0 + c1*x + c2*x^2 + ...

    """
    V = vandermonde_matrix(x_data)
    return solve_linear_system(V, y_data)


def main():
    # Example: interpolate f(x) = sin(x)
    import math

    x_data = [0, 0.5, 1.0, 1.5, 2.0]
    y_data = [math.sin(x) for x in x_data]

    # Evaluate at x = pi/4
    x = math.pi / 4
    y = vandermonde(x_data, y_data, x)
    print(f"Interpolated value at x = {x:.6f}: {y:.6f}")
    print(f"Actual value: {math.sin(x):.6f}")

    # Display polynomial coefficients
    coeffs = get_polynomial_coefficients(x_data, y_data)
    print("\nPolynomial coefficients:")
    for i, c in enumerate(coeffs):
        print(f"  c{i} = {c:.6f}")


if __name__ == "__main__":
    main()
