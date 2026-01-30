def givens_rotation(a, b):
    """
    Compute Givens rotation parameters c and s.

    The Givens rotation matrix G = [[c, s], [-s, c]] rotates the
    vector [a, b] to [r, 0] where r = sqrt(a^2 + b^2).

    Parameters
    ----------
    a : float
        First element.
    b : float
        Second element.

    Returns
    -------
    tuple of (float, float)
        The cosine c and sine s of the rotation.

    """
    if abs(b) < 1e-10:
        c = 1.0
        s = 0.0
    elif abs(a) < abs(b):
        tau = -a / b
        s = 1.0 / (1 + tau**2)**0.5
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / (1 + tau**2)**0.5
        s = c * tau

    return c, s


def givens_matrix(n, i, j, c, s):
    """
    Construct a Givens rotation matrix.

    The Givens matrix is an identity matrix with modified elements at
    positions (i,i), (i,j), (j,i), (j,j).

    Parameters
    ----------
    n : int
        Size of the matrix.
    i : int
        First row/column index.
    j : int
        Second row/column index.
    c : float
        Cosine of rotation angle.
    s : float
        Sine of rotation angle.

    Returns
    -------
    list of list of float
        The Givens rotation matrix.

    """
    G = [[1.0 if k == l else 0.0 for l in range(n)] for k in range(n)]

    G[i][i] = c
    G[i][j] = s
    G[j][i] = -s
    G[j][j] = c

    return G


def apply_givens_left(A, i, j, c, s):
    """
    Apply Givens rotation to matrix A from the left: G * A.

    This modifies rows i and j of A.

    Parameters
    ----------
    A : list of list of float
        The matrix to transform.
    i : int
        First row index.
    j : int
        Second row index.
    c : float
        Cosine of rotation.
    s : float
        Sine of rotation.

    Returns
    -------
    list of list of float
        The transformed matrix.

    """
    A_new = [row[:] for row in A]
    n = len(A[0]) if len(A) > 0 else 0

    for k in range(n):
        temp1 = c * A[i][k] + s * A[j][k]
        temp2 = -s * A[i][k] + c * A[j][k]
        A_new[i][k] = temp1
        A_new[j][k] = temp2

    return A_new


def apply_givens_right(A, i, j, c, s):
    """
    Apply Givens rotation to matrix A from the right: A * G^T.

    This modifies columns i and j of A.

    Parameters
    ----------
    A : list of list of float
        The matrix to transform.
    i : int
        First column index.
    j : int
        Second column index.
    c : float
        Cosine of rotation.
    s : float
        Sine of rotation.

    Returns
    -------
    list of list of float
        The transformed matrix.

    """
    A_new = [row[:] for row in A]
    m = len(A)

    for k in range(m):
        temp1 = c * A[k][i] - s * A[k][j]
        temp2 = s * A[k][i] + c * A[k][j]
        A_new[k][i] = temp1
        A_new[k][j] = temp2

    return A_new


def qr_givens(A):
    """
    Compute QR decomposition using Givens rotations.

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
    R = [row[:] for row in A]
    Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

    # Zero out elements below diagonal
    for j in range(n):
        for i in range(min(j + 1, m - 1), -1, -1):
            if i + 1 < m and abs(R[i + 1][j]) > 1e-10:
                # Compute Givens rotation to zero R[i+1, j]
                c, s = givens_rotation(R[i][j], R[i + 1][j])

                # Apply to R
                R = apply_givens_left(R, i, i + 1, c, s)

                # Apply to Q^T (accumulate)
                Q = apply_givens_right(Q, i, i + 1, c, s)

    return Q, R


def main():
    # Example 1: Simple Givens rotation
    a, b = 3.0, 4.0
    c, s = givens_rotation(a, b)

    print(f"Rotating vector [{a}, {b}]")
    print(f"c = {c:.6f}, s = {s:.6f}")

    # Apply rotation
    r1 = c * a + s * b
    r2 = -s * a + c * b
    print(f"Result: [{r1:.6f}, {r2:.6f}]")
    print(f"(Second element should be ~0)")

    # Example 2: Givens matrix
    G = givens_matrix(3, 0, 1, c, s)
    print("\nGivens matrix G:")
    for row in G:
        print([f"{val:.6f}" for val in row])

    # Example 3: QR decomposition
    A = [[6, 5, 0], [5, 1, 4], [0, 4, 3]]

    print("\n\nMatrix A:")
    for row in A:
        print(row)

    Q, R = qr_givens(A)

    print("\nQ (orthogonal):")
    for row in Q:
        print([f"{val:.6f}" for val in row])

    print("\nR (upper triangular):")
    for row in R:
        print([f"{val:.6f}" for val in row])

    # Verify: Q * R should equal A
    m = len(A)
    n = len(A[0])
    QR = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(m):
                QR[i][j] += Q[i][k] * R[k][j]

    print("\nVerification Q*R:")
    for row in QR:
        print([f"{val:.6f}" for val in row])


if __name__ == "__main__":
    main()
