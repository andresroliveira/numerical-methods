def householder_vector(x):
    """
    Compute the Householder vector for a given vector x.

    The Householder vector v is used to construct a reflection that
    zeros out all elements of x below the first element.

    Parameters
    ----------
    x : list of float
        The input vector.

    Returns
    -------
    list of float
        The Householder vector v.

    """
    v = x[:]

    # Compute norm of x
    norm_x = sum(xi**2 for xi in x) ** 0.5

    # Set sign to avoid cancellation
    sign = 1 if x[0] >= 0 else -1

    # First element of v
    v[0] = x[0] + sign * norm_x

    # Normalize v
    norm_v = sum(vi**2 for vi in v) ** 0.5
    if norm_v > 1e-10:
        v = [vi / norm_v for vi in v]

    return v


def householder_matrix(v):
    """
    Construct the Householder reflection matrix H = I - 2*v*v^T.

    Parameters
    ----------
    v : list of float
        The Householder vector.

    Returns
    -------
    list of list of float
        The Householder matrix H.

    """
    n = len(v)
    H = [[0.0] * n for _ in range(n)]

    # H = I - 2*v*v^T
    for i in range(n):
        for j in range(n):
            H[i][j] = -2 * v[i] * v[j]
            if i == j:
                H[i][j] += 1.0

    return H


def apply_householder(A, v, k=0):
    """
    Apply Householder reflection to matrix A starting at row/column k.

    Computes A = H * A where H = I - 2*v*v^T.

    Parameters
    ----------
    A : list of list of float
        The matrix to transform.
    v : list of float
        The Householder vector.
    k : int, optional
        Starting row/column index. Default is 0.

    Returns
    -------
    list of list of float
        The transformed matrix.

    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    A_new = [row[:] for row in A]

    # Compute H * A
    for i in range(k, m):
        for j in range(n):
            val = A[i][j]
            for row_idx in range(k, m):
                val -= 2 * v[i - k] * v[row_idx - k] * A[row_idx][j]
            A_new[i][j] = val

    return A_new


def qr_householder(A):
    """
    Compute QR decomposition using Householder reflections.

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

    for k in range(min(m - 1, n)):
        # Extract column k from row k onwards
        x = [R[i][k] for i in range(k, m)]

        # Compute Householder vector
        v = householder_vector(x)

        # Apply to R
        for j in range(n):
            # Compute (I - 2vv^T) * R[:, j] for rows k:m
            col = [R[i][j] for i in range(k, m)]
            dot = sum(v[i] * col[i] for i in range(len(v)))
            for i in range(k, m):
                R[i][j] -= 2 * v[i - k] * dot

        # Apply to Q (accumulate transformations)
        for j in range(m):
            col = [Q[i][j] for i in range(k, m)]
            dot = sum(v[i] * col[i] for i in range(len(v)))
            for i in range(k, m):
                Q[i][j] -= 2 * v[i - k] * dot

    # Transpose Q (since we accumulated Q^T)
    Q_T = [[Q[j][i] for j in range(m)] for i in range(m)]

    return Q_T, R


def main():
    # Example 1: Householder vector and matrix
    x = [3, 4, 0]
    print("Vector x:", x)

    v = householder_vector(x)
    print(f"Householder vector v: {[f'{vi:.6f}' for vi in v]}")

    H = householder_matrix(v)
    print("\nHouseholder matrix H:")
    for row in H:
        print([f"{val:.6f}" for val in row])

    # Apply H to x
    Hx = [sum(H[i][j] * x[j] for j in range(len(x))) for i in range(len(H))]
    print(f"\nH*x = {[f'{val:.6f}' for val in Hx]}")
    print("(Should zero out elements except first)")

    # Example 2: QR decomposition
    A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]

    print("\n\nMatrix A:")
    for row in A:
        print(row)

    Q, R = qr_householder(A)

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
