def cholesky(A):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix.

    For a symmetric positive definite matrix A, finds a lower triangular
    matrix L such that A = L * L^T.

    Parameters
    ----------
    A : list of list of float
        A symmetric positive definite matrix.

    Returns
    -------
    list of list of float
        The lower triangular matrix L.

    Raises
    ------
    ValueError
        If the matrix is not positive definite.

    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal elements
                sum_sq = sum(L[i][k]**2 for k in range(j))
                val = A[i][i] - sum_sq
                if val <= 0:
                    raise ValueError(
                        "Matrix is not positive definite at diagonal element"
                        f" ({i},{i})")
                L[i][j] = val**0.5
            else:
                # Off-diagonal elements
                sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - sum_prod) / L[j][j]

    return L


def solve_cholesky(A, b):
    """
    Solve the system Ax = b using Cholesky decomposition.

    Parameters
    ----------
    A : list of list of float
        A symmetric positive definite matrix.
    b : list of float
        The right-hand side vector.

    Returns
    -------
    list of float
        The solution vector x.

    """
    # Decompose A = L * L^T
    L = cholesky(A)
    n = len(b)

    # Solve L * y = b (forward substitution)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
        y[i] /= L[i][i]

    # Solve L^T * x = y (backward substitution)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] /= L[i][i]

    return x


def is_positive_definite(A, tol=1e-10):
    """
    Check if a matrix is symmetric positive definite.

    Parameters
    ----------
    A : list of list of float
        The matrix to check.
    tol : float, optional
        Tolerance for symmetry check. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix is symmetric positive definite, False otherwise.

    """
    n = len(A)

    # Check symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False

    # Try Cholesky decomposition
    try:
        cholesky(A)
        return True
    except ValueError:
        return False


def main():
    # Example 1: Simple 3x3 positive definite matrix
    A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]

    print("Matrix A:")
    for row in A:
        print(row)

    L = cholesky(A)
    print("\nCholesky factor L:")
    for row in L:
        print([f"{x:.4f}" for x in row])

    # Verify: compute L * L^T
    n = len(A)
    LLT = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            LLT[i][j] = sum(L[i][k] * L[j][k] for k in range(n))

    print("\nVerification L * L^T:")
    for row in LLT:
        print([f"{x:.4f}" for x in row])

    # Example 2: Solve a system
    A2 = [[4, 2], [2, 3]]
    b = [8, 7]

    print("\n\nSolving Ax = b:")
    print(f"A = {A2}")
    print(f"b = {b}")

    x = solve_cholesky(A2, b)
    print(f"x = {[f'{val:.4f}' for val in x]}")

    # Verify
    result = [
        sum(A2[i][j] * x[j] for j in range(len(x))) for i in range(len(A2))
    ]
    print(f"Verification Ax = {[f'{val:.4f}' for val in result]}")

    # Example 3: Check if matrix is positive definite
    A3 = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    print(f"\n\nIs matrix positive definite? {is_positive_definite(A3)}")


if __name__ == "__main__":
    main()
