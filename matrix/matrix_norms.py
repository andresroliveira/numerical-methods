def matrix_norm_1(A):
    """
    Compute the 1-norm (column-sum norm) of a matrix.

    The 1-norm is the maximum absolute column sum:
    ||A||_1 = max_j sum_i |a_ij|

    Parameters
    ----------
    A : list of list of float
        The matrix.

    Returns
    -------
    float
        The 1-norm of the matrix.

    """
    n = len(A)
    m = len(A[0]) if n > 0 else 0

    max_col_sum = 0.0
    for j in range(m):
        col_sum = sum(abs(A[i][j]) for i in range(n))
        max_col_sum = max(max_col_sum, col_sum)

    return max_col_sum


def matrix_norm_inf(A):
    """
    Compute the infinity-norm (row-sum norm) of a matrix.

    The infinity-norm is the maximum absolute row sum:
    ||A||_∞ = max_i sum_j |a_ij|

    Parameters
    ----------
    A : list of list of float
        The matrix.

    Returns
    -------
    float
        The infinity-norm of the matrix.

    """
    max_row_sum = 0.0
    for row in A:
        row_sum = sum(abs(x) for x in row)
        max_row_sum = max(max_row_sum, row_sum)

    return max_row_sum


def matrix_norm_frobenius(A):
    """
    Compute the Frobenius norm of a matrix.

    The Frobenius norm is:
    ||A||_F = sqrt(sum_i sum_j |a_ij|^2)

    Parameters
    ----------
    A : list of list of float
        The matrix.

    Returns
    -------
    float
        The Frobenius norm of the matrix.

    """
    sum_sq = 0.0
    for row in A:
        sum_sq += sum(x**2 for x in row)

    return sum_sq**0.5


def vector_norm_2(v):
    """
    Compute the Euclidean (2-norm) of a vector.

    Parameters
    ----------
    v : list of float
        The vector.

    Returns
    -------
    float
        The 2-norm of the vector.

    """
    return sum(x**2 for x in v)**0.5


def vector_norm_1(v):
    """
    Compute the 1-norm of a vector.

    Parameters
    ----------
    v : list of float
        The vector.

    Returns
    -------
    float
        The 1-norm of the vector.

    """
    return sum(abs(x) for x in v)


def vector_norm_inf(v):
    """
    Compute the infinity-norm of a vector.

    Parameters
    ----------
    v : list of float
        The vector.

    Returns
    -------
    float
        The infinity-norm of the vector.

    """
    return max(abs(x) for x in v) if v else 0.0


def condition_number_estimate(A, norm="2"):
    """
    Estimate the condition number of a matrix.

    The condition number is cond(A) = ||A|| * ||A^-1||.
    A high condition number indicates ill-conditioning.

    This is a simplified estimation. For exact computation,
    one would need to compute the inverse explicitly or use SVD.

    Parameters
    ----------
    A : list of list of float
        The matrix.
    norm : str, optional
        The norm to use: "1", "inf", or "fro". Default is "2" (uses Frobenius as approximation).

    Returns
    -------
    float
        An estimate of the condition number.

    """
    # This is a placeholder for demonstration
    # True condition number requires computing A^-1 or using eigenvalues/singular values
    if norm == "1":
        return matrix_norm_1(A)  # Simplified
    elif norm == "inf":
        return matrix_norm_inf(A)  # Simplified
    else:
        return matrix_norm_frobenius(A)  # Simplified


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
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C


def main():
    # Example matrices
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    B = [[2, -1], [0, 3]]

    print("Matrix A:")
    for row in A:
        print(row)

    print(f"\n||A||_1 (column-sum):     {matrix_norm_1(A):.4f}")
    print(f"||A||_∞ (row-sum):        {matrix_norm_inf(A):.4f}")
    print(f"||A||_F (Frobenius):      {matrix_norm_frobenius(A):.4f}")

    print("\n\nMatrix B:")
    for row in B:
        print(row)

    print(f"\n||B||_1 (column-sum):     {matrix_norm_1(B):.4f}")
    print(f"||B||_∞ (row-sum):        {matrix_norm_inf(B):.4f}")
    print(f"||B||_F (Frobenius):      {matrix_norm_frobenius(B):.4f}")

    # Vector norms
    v = [3, -4, 0, 5]
    print(f"\n\nVector v = {v}")
    print(f"||v||_1:   {vector_norm_1(v):.4f}")
    print(f"||v||_2:   {vector_norm_2(v):.4f}")
    print(f"||v||_∞:   {vector_norm_inf(v):.4f}")

    # Well-conditioned vs ill-conditioned
    A_well = [[2, 0], [0, 2]]
    A_ill = [[1, 1], [1, 1.0001]]

    print("\n\nWell-conditioned matrix:")
    for row in A_well:
        print(row)
    print(f"Frobenius norm: {matrix_norm_frobenius(A_well):.4f}")

    print("\nIll-conditioned matrix:")
    for row in A_ill:
        print(row)
    print(f"Frobenius norm: {matrix_norm_frobenius(A_ill):.4f}")


if __name__ == "__main__":
    main()
