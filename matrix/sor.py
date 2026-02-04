def sor(A, b, omega, x=None, tol=1e-6, max_iter=1000):
    """
    Solve the linear system Ax = b using Successive Over-Relaxation (SOR).

    SOR is a generalization of Gauss-Seidel with a relaxation parameter omega.
    - omega = 1: reduces to Gauss-Seidel
    - 0 < omega < 1: under-relaxation
    - 1 < omega < 2: over-relaxation (typically faster convergence)

    Parameters
    ----------
    A : list of list of float
        The matrix A in the system Ax = b.
    b : list of float
        The vector b in the system Ax = b.
    omega : float
        The relaxation parameter (typically 1 < omega < 2).
    x : list of float, optional
        The initial guess for the solution. Default is zero vector.
    tol : float, optional
        The tolerance for the stopping criterion. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. Default is 1000.

    Returns
    -------
    list of float
        The solution to the system Ax = b.

    """
    n = len(A)
    if x is None:
        x = [0.0] * n

    for iteration in range(max_iter):
        x_old = x[:]

        for i in range(n):
            # Compute sum of A[i,j] * x[j] for j < i (already updated)
            sum1 = sum(A[i][j] * x[j] for j in range(i))

            # Compute sum of A[i,j] * x[j] for j > i (not yet updated)
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))

            # SOR update
            x_gauss_seidel = (b[i] - sum1 - sum2) / A[i][i] if A[i][i] != 0 else 0
            x[i] = (1 - omega) * x_old[i] + omega * x_gauss_seidel

        # Check convergence
        if all(abs(x[i] - x_old[i]) < tol for i in range(n)):
            return x

    return x


def optimal_omega_estimate(A):
    """
    Estimate the optimal relaxation parameter for SOR.

    This is a rough estimate based on the spectral radius of the
    Jacobi iteration matrix. Works well for certain types of matrices
    (e.g., those arising from finite difference discretizations).

    Parameters
    ----------
    A : list of list of float
        The coefficient matrix.

    Returns
    -------
    float
        An estimated optimal omega value.

    """
    # For many problems, omega ≈ 1.5 to 1.9 works well
    # A more precise estimate requires computing the Jacobi spectral radius
    # For simplicity, we return a common value
    return 1.5


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
    tol : float, optional
        The tolerance for the verification. Default is 1e-6.

    Returns
    -------
    bool
        True if the solution is correct, False otherwise.

    """
    n = len(A)
    return all(
        abs(sum(A[i][j] * x[j] for j in range(n)) - b[i]) < tol for i in range(n)
    )


def main():
    # Example 1: Simple 3x3 system
    A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
    b = [1, 4, 3]

    print("Solving Ax = b using SOR")
    print(f"A = {A}")
    print(f"b = {b}")

    # Try different omega values
    omegas = [1.0, 1.2, 1.5, 1.8]

    for omega in omegas:
        x = sor(A, b, omega, tol=1e-8)
        print(f"\nω = {omega}")
        print(f"Solution: {[f'{val:.6f}' for val in x]}")
        print(f"Converged: {verify(A, b, x, tol=1e-6)}")

    # Example 2: Larger strictly diagonally dominant system
    A2 = [[10, 2, 1], [1, 5, 1], [2, 3, 10]]
    b2 = [1, 0, 3]

    print("\n\nSolving larger system:")
    print(f"A = {A2}")
    print(f"b = {b2}")

    omega_opt = 1.5
    x2 = sor(A2, b2, omega_opt, tol=1e-8)
    print(f"\nω = {omega_opt}")
    print(f"Solution: {[f'{val:.6f}' for val in x2]}")
    print(f"Verification: {verify(A2, b2, x2)}")

    # Compare with Gauss-Seidel (omega = 1)
    x_gs = sor(A2, b2, 1.0, tol=1e-8)
    print("\nGauss-Seidel (ω = 1.0):")
    print(f"Solution: {[f'{val:.6f}' for val in x_gs]}")


if __name__ == "__main__":
    main()
