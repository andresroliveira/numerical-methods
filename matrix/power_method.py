def power_method(A, tol=1e-6, max_iter=1000):
    """
    Find the dominant eigenvalue and eigenvector using the power method.

    Parameters
    ----------
    A : list of list of float
        The matrix for which to find eigenvalues.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    tuple of (float, list of float)
        The dominant eigenvalue and corresponding eigenvector.

    """
    n = len(A)

    # Initial guess (normalized random vector)
    v = [1.0] * n
    norm = sum(x**2 for x in v)**0.5
    v = [x / norm for x in v]

    eigenvalue = 0.0

    for _ in range(max_iter):
        # Multiply A * v
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]

        # Compute eigenvalue (Rayleigh quotient)
        eigenvalue_new = sum(v[i] * Av[i] for i in range(n))

        # Normalize
        norm = sum(x**2 for x in Av)**0.5
        v_new = [x / norm for x in Av]

        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tol:
            return eigenvalue_new, v_new

        eigenvalue = eigenvalue_new
        v = v_new

    return eigenvalue, v


def inverse_power_method(A, tol=1e-6, max_iter=1000):
    """
    Find the smallest eigenvalue and eigenvector using inverse power method.

    Parameters
    ----------
    A : list of list of float
        The matrix for which to find eigenvalues.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    tuple of (float, list of float)
        The smallest eigenvalue and corresponding eigenvector.

    """
    n = len(A)

    # Initial guess
    v = [1.0] * n
    norm = sum(x**2 for x in v)**0.5
    v = [x / norm for x in v]

    eigenvalue = 0.0

    for _ in range(max_iter):
        # Solve A * w = v
        w = solve_linear_system(A, v)

        # Compute eigenvalue
        eigenvalue_new = sum(v[i] * w[i] for i in range(n))

        # Normalize
        norm = sum(x**2 for x in w)**0.5
        v_new = [x / norm for x in w]

        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tol:
            return 1.0 / eigenvalue_new, v_new

        eigenvalue = eigenvalue_new
        v = v_new

    return 1.0 / eigenvalue, v


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
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i
        M[k], M[max_row] = M[max_row], M[k]

        # Elimination
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


def rayleigh_quotient(A, v):
    """
    Compute the Rayleigh quotient for a matrix and vector.

    The Rayleigh quotient is: R(A, v) = (v^T * A * v) / (v^T * v)

    Parameters
    ----------
    A : list of list of float
        The matrix.
    v : list of float
        The vector.

    Returns
    -------
    float
        The Rayleigh quotient.

    """
    n = len(A)
    Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
    numerator = sum(v[i] * Av[i] for i in range(n))
    denominator = sum(v[i]**2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0


def main():
    # Example: Find eigenvalues of a 3x3 matrix
    A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]

    print("Matrix A:")
    for row in A:
        print(row)

    # Find dominant eigenvalue
    eigenvalue_max, eigenvector_max = power_method(A, tol=1e-8)
    print(f"\nDominant eigenvalue: {eigenvalue_max:.6f}")
    print(f"Eigenvector: {[f'{x:.6f}' for x in eigenvector_max]}")

    # Verify: A * v should equal λ * v
    Av = [
        sum(A[i][j] * eigenvector_max[j] for j in range(len(A)))
        for i in range(len(A))
    ]
    lambda_v = [eigenvalue_max * x for x in eigenvector_max]
    print(f"\nVerification:")
    print(f"A*v = {[f'{x:.6f}' for x in Av]}")
    print(f"λ*v = {[f'{x:.6f}' for x in lambda_v]}")

    # Find smallest eigenvalue
    eigenvalue_min, eigenvector_min = inverse_power_method(A, tol=1e-8)
    print(f"\nSmallest eigenvalue: {eigenvalue_min:.6f}")
    print(f"Eigenvector: {[f'{x:.6f}' for x in eigenvector_min]}")

    # Example 2: Simple 2x2 matrix
    A2 = [[3, 1], [1, 3]]
    print("\n\nMatrix A2:")
    for row in A2:
        print(row)

    eigenvalue, eigenvector = power_method(A2)
    print(f"\nDominant eigenvalue: {eigenvalue:.6f}")
    print(f"Eigenvector: {[f'{x:.6f}' for x in eigenvector]}")
    print("(Exact eigenvalues: 4 and 2)")


if __name__ == "__main__":
    main()
