def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.

    Args:
        A: A 2D list representing a matrix.
        B: A 2D list representing a matrix.

    Returns:
        A 2D list representing the product of A and B.
    """

    n = len(A)
    m = len(B[0])
    C = [[0.0] * m for i in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]

    return C


def qr(A):
    """
    Perform QR decomposition of a matrix.

    Args:
        A: A 2D list representing a matrix.

    Returns:
        Q: A 2D list representing the orthogonal matrix.
        R: A 2D list representing the upper triangular matrix.
    """

    n = len(A)
    m = len(A[0])
    Q = [[0.0] * m for i in range(n)]
    R = [[0.0] * m for i in range(m)]
    V = [0.0] * n

    for i in range(m):
        for j in range(n):
            Q[j][i] = A[j][i]

        for j in range(i):
            R[j][i] = 0.0
            for k in range(n):
                R[j][i] += Q[k][i] * Q[k][j]
            for k in range(n):
                Q[k][i] -= R[j][i] * Q[k][j]

        V[i] = 0.0
        for j in range(n):
            V[i] += Q[j][i] ** 2
        V[i] = V[i] ** 0.5

        for j in range(n):
            Q[j][i] /= V[i]

    for i in range(m):
        for j in range(i, m):
            R[i][j] = 0.0
            for k in range(n):
                R[i][j] += Q[k][i] * A[k][j]

    return Q, R


def main():
    A = [[1, 3, 4], [2, 1, 3], [4, 1, 2]]
    Q, R = qr(A)
    print("Q:", Q)
    print("R:", R)
    print("QR:", matrix_multiply(Q, R))
    print("A:", A)


if __name__ == "__main__":
    main()
