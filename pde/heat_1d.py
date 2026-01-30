"""
1D Heat Equation: ∂u/∂t = α ∂²u/∂x²

Implements three finite difference schemes:
1. Explicit (Forward Euler in time, central difference in space)
2. Implicit (Backward Euler in time, central difference in space)
3. Crank-Nicolson (semi-implicit, 2nd order in time and space)
"""


def heat_explicit(alpha, x_min, x_max, nx, t_max, dt, u0, boundary_conditions):
    """
    Solve 1D heat equation using explicit (FTCS) method.

    ∂u/∂t = α ∂²u/∂x²

    Stability condition: r = α*dt/dx² ≤ 0.5

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient
    x_min : float
        Left boundary
    x_max : float
        Right boundary
    nx : int
        Number of spatial grid points
    t_max : float
        Final time
    dt : float
        Time step
    u0 : callable
        Initial condition u(x, 0) = u0(x)
    boundary_conditions : dict
        {'left': value, 'right': value} for Dirichlet conditions

    Returns
    -------
    x : list
        Spatial grid points
    t : list
        Time points
    u : list of lists
        Solution u[time_index][space_index]

    Raises
    ------
    ValueError
        If stability condition is violated

    Examples
    --------
    >>> def u0(x):
    ...     import math
    ...     return math.sin(math.pi * x)
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = heat_explicit(0.1, 0, 1, 21, 0.5, 0.001, u0, bc)
    >>> len(u) > 1  # Should produce multiple time steps
    True
    """
    # Spatial grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    # Check stability
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(
            f"Unstable! r = {r:.4f} > 0.5. Reduce dt or increase nx.")

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize solution
    u = []
    u_current = [u0(xi) for xi in x]
    u.append(list(u_current))

    # Time stepping
    for n in range(nt - 1):
        u_next = [0.0] * nx

        # Boundary conditions
        u_next[0] = boundary_conditions['left']
        u_next[-1] = boundary_conditions['right']

        # Interior points: u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
        for i in range(1, nx - 1):
            u_next[i] = u_current[i] + r * (
                u_current[i + 1] - 2 * u_current[i] + u_current[i - 1])

        u_current = u_next
        u.append(list(u_current))

    return x, t, u


def heat_implicit(alpha, x_min, x_max, nx, t_max, dt, u0, boundary_conditions):
    """
    Solve 1D heat equation using implicit (BTCS) method.

    Unconditionally stable. Requires solving tridiagonal system at each time step.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient
    x_min : float
        Left boundary
    x_max : float
        Right boundary
    nx : int
        Number of spatial grid points
    t_max : float
        Final time
    dt : float
        Time step
    u0 : callable
        Initial condition u(x, 0) = u0(x)
    boundary_conditions : dict
        {'left': value, 'right': value} for Dirichlet conditions

    Returns
    -------
    x : list
        Spatial grid points
    t : list
        Time points
    u : list of lists
        Solution u[time_index][space_index]

    Examples
    --------
    >>> def u0(x):
    ...     return x * (1 - x)
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = heat_implicit(0.1, 0, 1, 11, 0.5, 0.05, u0, bc)
    >>> u[-1][5] < u[0][5]  # Temperature at center decreases
    True
    """
    # Spatial grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    r = alpha * dt / (dx * dx)

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize solution
    u = []
    u_current = [u0(xi) for xi in x]
    u.append(list(u_current))

    # Tridiagonal matrix coefficients
    # -r*u[i-1] + (1+2r)*u[i] - r*u[i+1] = u_old[i]
    a = [-r] * (nx - 2)  # Lower diagonal
    b = [1 + 2 * r] * (nx - 2)  # Main diagonal
    c = [-r] * (nx - 2)  # Upper diagonal

    # Time stepping
    for n in range(nt - 1):
        # Right-hand side
        d = list(u_current[1:-1])

        # Apply boundary conditions
        d[0] += r * boundary_conditions['left']
        d[-1] += r * boundary_conditions['right']

        # Solve tridiagonal system
        u_interior = solve_tridiagonal(a, b, c, d)

        # Assemble solution
        u_next = [boundary_conditions['left']
                  ] + u_interior + [boundary_conditions['right']]

        u_current = u_next
        u.append(list(u_current))

    return x, t, u


def heat_crank_nicolson(alpha, x_min, x_max, nx, t_max, dt, u0,
                        boundary_conditions):
    """
    Solve 1D heat equation using Crank-Nicolson method.

    Semi-implicit, 2nd order in both time and space. Unconditionally stable.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient
    x_min : float
        Left boundary
    x_max : float
        Right boundary
    nx : int
        Number of spatial grid points
    t_max : float
        Final time
    dt : float
        Time step
    u0 : callable
        Initial condition u(x, 0) = u0(x)
    boundary_conditions : dict
        {'left': value, 'right': value} for Dirichlet conditions

    Returns
    -------
    x : list
        Spatial grid points
    t : list
        Time points
    u : list of lists
        Solution u[time_index][space_index]

    Examples
    --------
    >>> def u0(x):
    ...     import math
    ...     return math.exp(-100 * (x - 0.5)**2)
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = heat_crank_nicolson(0.1, 0, 1, 51, 1.0, 0.01, u0, bc)
    >>> max(u[-1]) < max(u[0])  # Heat dissipates
    True
    """
    # Spatial grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    r = alpha * dt / (dx * dx)

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize solution
    u = []
    u_current = [u0(xi) for xi in x]
    u.append(list(u_current))

    # Tridiagonal matrix for implicit part
    # -r/2*u[i-1] + (1+r)*u[i] - r/2*u[i+1] = RHS
    a = [-r / 2] * (nx - 2)
    b = [1 + r] * (nx - 2)
    c = [-r / 2] * (nx - 2)

    # Time stepping
    for n in range(nt - 1):
        # Right-hand side from explicit part
        d = [0.0] * (nx - 2)

        for i in range(1, nx - 1):
            j = i - 1  # Index in interior array
            d[j] = (r / 2) * u_current[i - 1] + (1 - r) * u_current[i] + (
                r / 2) * u_current[i + 1]

        # Apply boundary conditions
        d[0] += (r / 2) * boundary_conditions['left']
        d[-1] += (r / 2) * boundary_conditions['right']

        # Solve tridiagonal system
        u_interior = solve_tridiagonal(a, b, c, d)

        # Assemble solution
        u_next = [boundary_conditions['left']
                  ] + u_interior + [boundary_conditions['right']]

        u_current = u_next
        u.append(list(u_current))

    return x, t, u


def solve_tridiagonal(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Parameters
    ----------
    a : list
        Lower diagonal (length n-1, a[0] not used)
    b : list
        Main diagonal (length n)
    c : list
        Upper diagonal (length n-1, c[-1] not used)
    d : list
        Right-hand side (length n)

    Returns
    -------
    x : list
        Solution vector
    """
    n = len(d)
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    x = [0.0] * n

    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    # Back substitution
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def main():
    """Example usage of 1D heat equation solvers."""
    import math

    print("=" * 70)
    print("1D Heat Equation: ∂u/∂t = α ∂²u/∂x²")
    print("=" * 70)

    # Example 1: Explicit method with sine initial condition
    print(
        "\n1. Explicit method: u(x,0) = sin(πx), boundaries u(0,t) = u(1,t) = 0"
    )

    alpha = 0.1

    def u0_sine(x):
        return math.sin(math.pi * x)

    bc = {'left': 0, 'right': 0}

    x, t, u_exp = heat_explicit(alpha, 0, 1, 21, 0.5, 0.001, u0_sine, bc)

    print(f"   Grid: {len(x)} points, Time steps: {len(t)}")
    print(
        f"   Stability parameter r = {alpha * 0.001 / (0.05**2):.4f} (must be ≤ 0.5)"
    )
    print(f"\n   x\t\tt=0\t\tt={t[-1]:.3f}")
    print("   " + "-" * 40)
    for i in [0, 5, 10, 15, 20]:
        print(f"   {x[i]:.2f}\t{u_exp[0][i]:.6f}\t{u_exp[-1][i]:.6f}")

    # Example 2: Compare all three methods
    print("\n2. Method comparison: u(x,0) = x(1-x)")

    def u0_parabola(x):
        return x * (1 - x)

    dt = 0.01
    nx = 21

    x, t_exp, u_exp = heat_explicit(alpha, 0, 1, nx, 0.2, dt, u0_parabola, bc)
    x, t_imp, u_imp = heat_implicit(alpha, 0, 1, nx, 0.2, dt, u0_parabola, bc)
    x, t_cn, u_cn = heat_crank_nicolson(alpha, 0, 1, nx, 0.2, dt, u0_parabola,
                                        bc)

    print(f"\n   Center point x = 0.5, t = {t_exp[-1]:.2f}")
    print(f"   Explicit:        u = {u_exp[-1][10]:.8f}")
    print(f"   Implicit:        u = {u_imp[-1][10]:.8f}")
    print(f"   Crank-Nicolson:  u = {u_cn[-1][10]:.8f}")

    # Example 3: Stability test
    print("\n3. Stability test: Explicit vs Implicit with large dt")

    dt_large = 0.05
    r = alpha * dt_large / ((1.0 / 20)**2)

    print(f"   Using dt = {dt_large}, r = {r:.4f}")

    if r > 0.5:
        print(f"   Explicit: UNSTABLE (r > 0.5)")
    else:
        x, t, u_exp_large = heat_explicit(alpha, 0, 1, 21, 0.2, dt_large,
                                          u0_parabola, bc)
        print(f"   Explicit: Stable, u(0.5, 0.2) = {u_exp_large[-1][10]:.6f}")

    x, t, u_imp_large = heat_implicit(alpha, 0, 1, 21, 0.2, dt_large,
                                      u0_parabola, bc)
    print(
        f"   Implicit: Always stable, u(0.5, 0.2) = {u_imp_large[-1][10]:.6f}")

    # Example 4: Heat dissipation
    print("\n4. Gaussian pulse dissipation: u(x,0) = exp(-100(x-0.5)²)")

    def u0_gaussian(x):
        return math.exp(-100 * (x - 0.5)**2)

    x, t, u_gauss = heat_crank_nicolson(alpha, 0, 1, 51, 0.5, 0.01,
                                        u0_gaussian, bc)

    print(f"\n   t\t\tMax u\t\tDecay")
    print("   " + "-" * 40)

    for i in [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, -1]:
        max_u = max(u_gauss[i])
        if i == 0:
            print(f"   {t[i]:.3f}\t{max_u:.6f}\t-")
        else:
            decay = (1 - max_u / max(u_gauss[0])) * 100
            print(f"   {t[i]:.3f}\t{max_u:.6f}\t{decay:.1f}%")

    print("\n" + "=" * 70)
    print("Notes:")
    print("  - Explicit: Simple but requires small dt (r ≤ 0.5)")
    print("  - Implicit: Unconditionally stable but requires solving system")
    print(
        "  - Crank-Nicolson: Best accuracy (2nd order), unconditionally stable"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
