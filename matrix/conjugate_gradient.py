def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=None):
    """
    Solve the system Ax = b using the conjugate gradient method.

    This method is suitable for symmetric positive definite matrices.

    Parameters
    ----------
    A : list of list of float
        A symmetric positive definite matrix.
    b : list of float
        The right-hand side vector.
    x0 : list of float, optional
        Initial guess. Default is zero vector.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is len(b).

    Returns
    -------
    list of float
        The approximate solution vector x.

    """
    n = len(b)

    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    if max_iter is None:
        max_iter = n

    # Compute initial residual r = b - A*x
    Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    r = [b[i] - Ax[i] for i in range(n)]

    # Initial search direction
    p = r[:]

    # Initial residual norm squared
    rs_old = sum(r[i]**2 for i in range(n))

    for iteration in range(max_iter):
        # Check convergence
        if rs_old**0.5 < tol:
            break

        # Compute A*p
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]

        # Compute step size alpha
        pAp = sum(p[i] * Ap[i] for i in range(n))
        alpha = rs_old / pAp if pAp != 0 else 0

        # Update solution: x = x + alpha * p
        x = [x[i] + alpha * p[i] for i in range(n)]

        # Update residual: r = r - alpha * A*p
        r = [r[i] - alpha * Ap[i] for i in range(n)]

        # Compute new residual norm squared
        rs_new = sum(r[i]**2 for i in range(n))

        # Compute beta
        beta = rs_new / rs_old if rs_old != 0 else 0

        # Update search direction: p = r + beta * p
        p = [r[i] + beta * p[i] for i in range(n)]

        # Update residual norm
        rs_old = rs_new

    return x


def preconditioned_conjugate_gradient(A,
                                      b,
                                      M_inv,
                                      x0=None,
                                      tol=1e-6,
                                      max_iter=None):
    """
    Solve Ax = b using preconditioned conjugate gradient method.

    Parameters
    ----------
    A : list of list of float
        A symmetric positive definite matrix.
    b : list of float
        The right-hand side vector.
    M_inv : list of list of float
        Preconditioner (inverse of M, where M approximates A).
    x0 : list of float, optional
        Initial guess. Default is zero vector.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is len(b).

    Returns
    -------
    list of float
        The approximate solution vector x.

    """
    n = len(b)

    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    if max_iter is None:
        max_iter = n

    # Compute initial residual r = b - A*x
    Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    r = [b[i] - Ax[i] for i in range(n)]

    # Apply preconditioner: z = M_inv * r
    z = [sum(M_inv[i][j] * r[j] for j in range(n)) for i in range(n)]

    # Initial search direction
    p = z[:]

    # Initial value
    rz_old = sum(r[i] * z[i] for i in range(n))

    for iteration in range(max_iter):
        # Check convergence
        if sum(r[i]**2 for i in range(n))**0.5 < tol:
            break

        # Compute A*p
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]

        # Compute step size alpha
        pAp = sum(p[i] * Ap[i] for i in range(n))
        alpha = rz_old / pAp if pAp != 0 else 0

        # Update solution
        x = [x[i] + alpha * p[i] for i in range(n)]

        # Update residual
        r = [r[i] - alpha * Ap[i] for i in range(n)]

        # Apply preconditioner
        z = [sum(M_inv[i][j] * r[j] for j in range(n)) for i in range(n)]

        # Compute new value
        rz_new = sum(r[i] * z[i] for i in range(n))

        # Compute beta
        beta = rz_new / rz_old if rz_old != 0 else 0

        # Update search direction
        p = [z[i] + beta * p[i] for i in range(n)]

        # Update value
        rz_old = rz_new

    return x


def main():
    # Example 1: Simple positive definite system
    A = [[4, 1], [1, 3]]
    b = [1, 2]

    print("Solving Ax = b with Conjugate Gradient")
    print(f"A = {A}")
    print(f"b = {b}")

    x = conjugate_gradient(A, b)
    print(f"\nSolution: x = {[f'{val:.6f}' for val in x]}")

    # Verify
    result = [
        sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))
    ]
    print(f"Verification Ax = {[f'{val:.6f}' for val in result]}")

    # Example 2: Larger system
    A2 = [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
    b2 = [1, 2, 3]

    print("\n\nSolving larger system:")
    print(f"A = {A2}")
    print(f"b = {b2}")

    x2 = conjugate_gradient(A2, b2, tol=1e-8)
    print(f"\nSolution: x = {[f'{val:.6f}' for val in x2]}")

    # Verify
    result2 = [
        sum(A2[i][j] * x2[j] for j in range(len(x2))) for i in range(len(A2))
    ]
    print(f"Verification Ax = {[f'{val:.6f}' for val in result2]}")

    # Example 3: Compare with exact solution
    # For A = [[4,1],[1,3]], b = [1,2]
    # Exact solution: x = [1/11, 7/11] ≈ [0.090909, 0.636364]
    print(f"\nExact solution: [0.090909, 0.636364]")
    print(f"CG solution:    {[f'{val:.6f}' for val in x]}")
    print(f"Error: {[(abs(x[i] - [1/11, 7/11][i]))for i in range(2)]}")


if __name__ == "__main__":
    main()
