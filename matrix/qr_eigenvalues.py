def qr_algorithm(A, tol=1e-8, max_iter=1000):
    """
    Compute eigenvalues of a matrix using the QR algorithm.

    The QR algorithm repeatedly performs QR decomposition and
    updates A = R*Q. The diagonal elements converge to eigenvalues.

    Parameters
    ----------
    A : list of list of float
        A square matrix.
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    list of float
        The eigenvalues (diagonal elements after convergence).

    """
    n = len(A)
    A_k = [row[:] for row in A]

    for iteration in range(max_iter):
        # QR decomposition
        Q, R = qr_decomposition_gram_schmidt(A_k)

        # Update A_k = R * Q
        A_k = matrix_multiply(R, Q)

        # Check convergence (off-diagonal elements should be near zero)
        off_diag_norm = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_norm += abs(A_k[i][j])

        if off_diag_norm < tol:
            break

    # Extract eigenvalues from diagonal
    eigenvalues = [A_k[i][i] for i in range(n)]

    return eigenvalues


def qr_decomposition_gram_schmidt(A):
    """
    Compute QR decomposition using Gram-Schmidt orthogonalization.

    Parameters
    ----------
    A : list of list of float
        The matrix to decompose.

    Returns
    -------
    tuple of (list of list of float, list of list of float)
        The orthogonal matrix Q and upper triangular matrix R.

    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    Q = [[0.0] * n for _ in range(m)]
    R = [[0.0] * n for _ in range(n)]

    for j in range(n):
        # Copy column j
        v = [A[i][j] for i in range(m)]

        # Orthogonalize against previous columns
        for i in range(j):
            R[i][j] = sum(Q[k][i] * A[k][j] for k in range(m))
            for k in range(m):
                v[k] -= R[i][j] * Q[k][i]

        # Normalize
        R[j][j] = sum(v[k] ** 2 for k in range(m)) ** 0.5

        if R[j][j] > 1e-10:
            for k in range(m):
                Q[k][j] = v[k] / R[j][j]

    return Q, R


def matrix_multiply(A, B):
    """
    Multiply two matrices.

    Parameters
    ----------
    A : list of list of float
        First matrix.
    B : list of list of float
        Second matrix.

    Returns
    -------
    list of list of float
        The product A * B.

    """
    m = len(A)
    n = len(B[0]) if len(B) > 0 else 0
    p = len(B)
    C = [[0.0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C


def qr_shifted(A, tol=1e-8, max_iter=1000):
    """
    Compute eigenvalues using QR algorithm with Wilkinson shift.

    The shift accelerates convergence by using the eigenvalue of
    the lower-right 2x2 submatrix closest to A[n-1,n-1].

    Parameters
    ----------
    A : list of list of float
        A square matrix.
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    list of float
        The eigenvalues.

    """
    n = len(A)
    A_k = [row[:] for row in A]
    eigenvalues = []

    while len(eigenvalues) < n and max_iter > 0:
        m = len(A_k)

        if m == 1:
            eigenvalues.append(A_k[0][0])
            break

        # Wilkinson shift
        delta = (A_k[m - 2][m - 2] - A_k[m - 1][m - 1]) / 2
        sign = 1 if delta >= 0 else -1
        mu = A_k[m - 1][m - 1] - sign * A_k[m - 1][m - 2] ** 2 / (
            abs(delta) + (delta**2 + A_k[m - 1][m - 2] ** 2) ** 0.5
        )

        # Shift: A_k - mu*I
        for i in range(m):
            A_k[i][i] -= mu

        # QR decomposition
        Q, R = qr_decomposition_gram_schmidt(A_k)

        # Update: A_k = R*Q + mu*I
        A_k = matrix_multiply(R, Q)
        for i in range(m):
            A_k[i][i] += mu

        # Check if last row is negligible
        if abs(A_k[m - 1][m - 2]) < tol:
            eigenvalues.append(A_k[m - 1][m - 1])
            # Deflate
            A_k = [row[:-1] for row in A_k[:-1]]

        max_iter -= 1

    return eigenvalues


def main():
    # Example 1: Simple symmetric matrix
    A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]

    print("Matrix A:")
    for row in A:
        print(row)

    eigenvalues = qr_algorithm(A, tol=1e-10)
    print("\nEigenvalues (QR algorithm):")
    for i, eig in enumerate(eigenvalues):
        print(f"  λ{i + 1} = {eig:.8f}")

    print("\n(Exact eigenvalues: 2-√2, 2, 2+√2)")
    print(f"(Approx: {2 - 2**0.5:.8f}, 2.0, {2 + 2**0.5:.8f})")

    # Example 2: Another symmetric matrix
    A2 = [[3, 1], [1, 3]]

    print("\n\nMatrix A2:")
    for row in A2:
        print(row)

    eigenvalues2 = qr_algorithm(A2)
    print("\nEigenvalues:")
    for i, eig in enumerate(eigenvalues2):
        print(f"  λ{i + 1} = {eig:.8f}")

    print("\n(Exact eigenvalues: 2, 4)")


if __name__ == "__main__":
    main()
