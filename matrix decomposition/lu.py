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


def lu(A):
    """
    Perform LU decomposition of a square matrix.

    Args:
        A: A 2D list representing a square matrix.

    Returns:
        L: A 2D list representing the lower triangular matrix.
        U: A 2D list representing the upper triangular matrix.
    """

    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = A[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]

    return L, U


def main():
    A = [[1, 3, 4], [2, 1, 3], [4, 1, 2]]
    L, U = lu(A)
    print("L:", L)
    print("U:", U)
    print("L*U:", matrix_multiply(L, U))
    print("A:", A)


if __name__ == "__main__":
    main()
