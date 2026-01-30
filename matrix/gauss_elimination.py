def gauss_elimination(A, b):
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
        The solution vector x.

    """
    n = len(A)
    # Create augmented matrix [A|b]
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot (row with maximum value in column k)
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i

        # Swap rows k and max_row
        M[k], M[max_row] = M[max_row], M[k]

        # Eliminate entries below pivot
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


def verify(A, b, x, tol=1e-6):
    """
    Verify the solution to the system Ax = b.

    Parameters
    ----------
    A : list of list of float
        The matrix A in the system Ax = b.
    b : list of float
        The vector b in the system Ax = b.
    x : list of float
        The solution to the system Ax = b.
    tol : float
        The tolerance for the verification.

    Returns
    -------
    bool
        True if the solution is correct, False otherwise.

    """
    n = len(A)
    return all(
        abs(sum(A[i][j] * x[j] for j in range(n)) - b[i]) < tol
        for i in range(n))


def main():
    # Example: solve a 3x3 system
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]

    x = gauss_elimination(A, b)
    print("Solution:", x)
    print("Verification:", verify(A, b, x))

    # Expected solution: [2, 3, -1]
    print("Expected: [2, 3, -1]")


if __name__ == "__main__":
    main()
