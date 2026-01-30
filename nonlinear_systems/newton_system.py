def newton_system(F, J, x0, tol=1e-6, max_iter=50):
    """
    Solve a system of nonlinear equations using Newton's method.

    The system is F(x) = 0, where F: R^n -> R^n.

    Parameters
    ----------
    F : function
        A function that takes a list x and returns a list F(x).
    J : function
        A function that takes a list x and returns the Jacobian matrix J(x).
        J[i][j] = ∂F_i/∂x_j
    x0 : list of float
        The initial guess for the solution.
    tol : float, optional
        The convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. Default is 50.

    Returns
    -------
    list of float
        The approximate solution to F(x) = 0.

    """
    x = x0[:]
    n = len(x)

    for iteration in range(max_iter):
        # Evaluate F(x) and J(x)
        F_val = F(x)
        J_val = J(x)

        # Solve J(x) * delta = -F(x)
        delta = solve_linear_system(J_val, [-f for f in F_val])

        # Update: x = x + delta
        x = [x[i] + delta[i] for i in range(n)]

        # Check convergence
        if max(abs(d) for d in delta) < tol:
            return x

    return x


def solve_linear_system(A, b):
    """
    Solve a linear system Ax = b using Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : list of list of float
        The coefficient matrix.
    b : list of float
        The right-hand side vector.

    Returns
    -------
    list of float
        The solution vector.

    """
    n = len(A)
    # Create augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i

        # Swap rows
        M[k], M[max_row] = M[max_row], M[k]

        # Eliminate column
        for i in range(k + 1, n):
            if M[k][k] != 0:
                factor = M[i][k] / M[k][k]
                for j in range(k, n + 1):
                    M[i][j] -= factor * M[k][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if M[i][i] != 0:
            x[i] /= M[i][i]

    return x


def main():
    # Example 1: Simple 2x2 system
    # x^2 + y^2 = 5
    # x*y = 2
    # Solution: (x, y) ≈ (2, 1) or (1, 2)

    def F(x):
        return [x[0]**2 + x[1]**2 - 5, x[0] * x[1] - 2]

    def J(x):
        return [[2 * x[0], 2 * x[1]], [x[1], x[0]]]

    x0 = [1.5, 1.5]
    solution = newton_system(F, J, x0)

    print("System:")
    print("  x^2 + y^2 = 5")
    print("  x*y = 2")
    print(f"\nSolution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
    print(f"Verification:")
    print(f"  x^2 + y^2 = {solution[0]**2 + solution[1]**2:.6f}")
    print(f"  x*y = {solution[0]*solution[1]:.6f}")

    # Example 2: 3x3 system
    # x^2 + y^2 + z^2 = 14
    # x*y*z = 8
    # x + y + z = 6
    # Solution: (x, y, z) = (2, 2, 2) is one solution

    def G(x):
        return [
            x[0]**2 + x[1]**2 + x[2]**2 - 14,
            x[0] * x[1] * x[2] - 8,
            x[0] + x[1] + x[2] - 6,
        ]

    def J_G(x):
        return [
            [2 * x[0], 2 * x[1], 2 * x[2]],
            [x[1] * x[2], x[0] * x[2], x[0] * x[1]],
            [1, 1, 1],
        ]

    x0 = [1.0, 2.0, 3.0]
    solution2 = newton_system(G, J_G, x0)

    print("\n\nSystem:")
    print("  x^2 + y^2 + z^2 = 14")
    print("  x*y*z = 8")
    print("  x + y + z = 6")
    print(
        f"\nSolution: x = {solution2[0]:.6f}, y = {solution2[1]:.6f}, z = {solution2[2]:.6f}"
    )
    print(f"Verification:")
    print(
        f"  x^2 + y^2 + z^2 = {solution2[0]**2 + solution2[1]**2 + solution2[2]**2:.6f}"
    )
    print(f"  x*y*z = {solution2[0]*solution2[1]*solution2[2]:.6f}")
    print(f"  x + y + z = {solution2[0] + solution2[1] + solution2[2]:.6f}")


if __name__ == "__main__":
    main()
