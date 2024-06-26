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
        abs(sum(A[i][j] * x[j] for j in range(n)) - b[i]) < tol for i in range(n)
    )


def jacobi(A, b, x=None, tol=1e-10, max_iter=1000):
    """
    Jacobi method for solving the linear system Ax = b.

    Parameters
    ----------
    A : list of list of float
        The matrix A in the system Ax = b.
    b : list of float
        The vector b in the system Ax = b.
    x : list of float
        The initial guess for the solution.
    tol : float
        The tolerance for the stopping criterion.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    list of float
        The solution to the system Ax = b.
    """
    n = len(A)
    if x is None:
        x = [0] * n
    x_new = x.copy()
    for _ in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            break
        x, x_new = x_new, x
    return x


def main():
    A = [[10, 2, 1], [1, 5, 1], [2, 3, 10]]
    b = [1, 0, 3]
    x = jacobi(A, b)
    print("Solution:", x)
    print("Verification:", verify(A, b, x))



if __name__ == "__main__":
    main()
