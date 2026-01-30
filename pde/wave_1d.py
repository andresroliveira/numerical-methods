"""
1D Wave Equation: ∂²u/∂t² = c² ∂²u/∂x²

Implements various explicit finite difference schemes:
- Standard explicit (centered differences)
- Lax-Friedrichs (dissipative, stable)
- Upwind (for advection)
- Lax-Wendroff (2nd order)

All methods require CFL condition for stability.
"""


def wave_explicit(c, x_min, x_max, nx, t_max, dt, u0, v0, boundary_conditions):
    """
    Solve 1D wave equation using standard explicit method.

    ∂²u/∂t² = c² ∂²u/∂x²

    CFL stability condition: c*dt/dx ≤ 1

    Parameters
    ----------
    c : float
        Wave speed
    x_min, x_max : float
        Spatial domain
    nx : int
        Number of spatial points
    t_max : float
        Final time
    dt : float
        Time step
    u0 : callable
        Initial displacement u(x, 0) = u0(x)
    v0 : callable
        Initial velocity ∂u/∂t(x, 0) = v0(x)
    boundary_conditions : dict
        {'left': value, 'right': value} for Dirichlet

    Returns
    -------
    x : list
        Spatial grid
    t : list
        Time points
    u : list of lists
        Solution u[time_index][space_index]

    Raises
    ------
    ValueError
        If CFL condition is violated

    Examples
    --------
    >>> import math
    >>> def u0(x):
    ...     return math.exp(-100 * (x - 0.5)**2)
    >>> def v0(x):
    ...     return 0
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = wave_explicit(1.0, 0, 1, 51, 1.0, 0.01, u0, v0, bc)
    >>> len(u) > 10  # Multiple time steps
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    # CFL number
    r = c * dt / dx
    if r > 1:
        raise ValueError(f"CFL condition violated: c*dt/dx = {r:.4f} > 1")

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize
    u = []
    u_curr = [u0(xi) for xi in x]
    u.append(list(u_curr))

    # First step: use Taylor expansion with initial velocity
    u_prev = [0.0] * nx
    for i in range(nx):
        u_prev[i] = u_curr[i] - dt * v0(x[i])

    # Time stepping
    r_sq = r * r

    for n in range(nt - 1):
        u_next = [0.0] * nx

        # Boundary conditions
        u_next[0] = boundary_conditions['left']
        u_next[-1] = boundary_conditions['right']

        # Interior: u_new = 2*u - u_old + r²*(u[i+1] - 2*u[i] + u[i-1])
        for i in range(1, nx - 1):
            u_next[i] = (2 * u_curr[i] - u_prev[i] + r_sq *
                         (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]))

        u_prev = u_curr
        u_curr = u_next
        u.append(list(u_curr))

    return x, t, u


def wave_lax_friedrichs(c, x_min, x_max, nx, t_max, dt, u0, v0,
                        boundary_conditions):
    """
    Solve 1D wave equation using Lax-Friedrichs scheme.

    More dissipative but more stable than standard explicit.

    Parameters
    ----------
    Same as wave_explicit

    Returns
    -------
    Same as wave_explicit

    Examples
    --------
    >>> import math
    >>> def u0(x):
    ...     return math.sin(2 * math.pi * x)
    >>> def v0(x):
    ...     return 0
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = wave_lax_friedrichs(1.0, 0, 1, 41, 2.0, 0.02, u0, v0, bc)
    >>> abs(max(u[-1])) < abs(max(u[0]))  # Some dissipation
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    r = c * dt / dx
    if r > 1:
        raise ValueError(f"CFL condition violated: c*dt/dx = {r:.4f} > 1")

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize
    u = []
    u_curr = [u0(xi) for xi in x]
    u.append(list(u_curr))

    # First step
    u_prev = [0.0] * nx
    for i in range(nx):
        u_prev[i] = u_curr[i] - dt * v0(x[i])

    # Time stepping
    r_sq = r * r

    for n in range(nt - 1):
        u_next = [0.0] * nx

        # Boundary conditions
        u_next[0] = boundary_conditions['left']
        u_next[-1] = boundary_conditions['right']

        # Lax-Friedrichs: includes averaging for stability
        for i in range(1, nx - 1):
            u_avg = 0.5 * (u_curr[i + 1] + u_curr[i - 1])
            u_next[i] = (2 * u_avg - u_prev[i] + r_sq *
                         (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]))

        u_prev = u_curr
        u_curr = u_next
        u.append(list(u_curr))

    return x, t, u


def wave_upwind(c, x_min, x_max, nx, t_max, dt, u0, boundary_conditions):
    """
    Solve 1D advection equation using upwind scheme.

    For ∂u/∂t + c ∂u/∂x = 0 (first-order wave equation)

    Parameters
    ----------
    c : float
        Wave speed (can be negative)
    x_min, x_max : float
        Spatial domain
    nx : int
        Number of spatial points
    t_max : float
        Final time
    dt : float
        Time step
    u0 : callable
        Initial condition u(x, 0) = u0(x)
    boundary_conditions : dict
        {'left': value, 'right': value}

    Returns
    -------
    x : list
        Spatial grid
    t : list
        Time points
    u : list of lists
        Solution

    Examples
    --------
    >>> def u0(x):
    ...     return 1 if 0.3 < x < 0.5 else 0
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = wave_upwind(1.0, 0, 1, 101, 0.2, 0.005, u0, bc)
    >>> max(u[-1]) > 0.5  # Pulse propagates
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    # CFL
    r = c * dt / dx
    if abs(r) > 1:
        raise ValueError(
            f"CFL condition violated: |c*dt/dx| = {abs(r):.4f} > 1")

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize
    u = []
    u_curr = [u0(xi) for xi in x]
    u.append(list(u_curr))

    # Time stepping
    for n in range(nt - 1):
        u_next = [0.0] * nx

        # Boundary conditions
        u_next[0] = boundary_conditions['left']
        u_next[-1] = boundary_conditions['right']

        # Upwind scheme
        if c > 0:
            # Backward difference
            for i in range(1, nx - 1):
                u_next[i] = u_curr[i] - r * (u_curr[i] - u_curr[i - 1])
        else:
            # Forward difference
            for i in range(1, nx - 1):
                u_next[i] = u_curr[i] - r * (u_curr[i + 1] - u_curr[i])

        u_curr = u_next
        u.append(list(u_curr))

    return x, t, u


def wave_lax_wendroff(c, x_min, x_max, nx, t_max, dt, u0, boundary_conditions):
    """
    Solve 1D advection equation using Lax-Wendroff scheme.

    Second-order accurate in space and time.
    For ∂u/∂t + c ∂u/∂x = 0

    Parameters
    ----------
    Same as wave_upwind

    Returns
    -------
    Same as wave_upwind

    Examples
    --------
    >>> import math
    >>> def u0(x):
    ...     return math.exp(-100 * (x - 0.3)**2)
    >>> bc = {'left': 0, 'right': 0}
    >>> x, t, u = wave_lax_wendroff(1.0, 0, 1, 101, 0.3, 0.005, u0, bc)
    >>> max(u[-1]) > 0.7  # Better preservation than upwind
    True
    """
    # Grid
    dx = (x_max - x_min) / (nx - 1)
    x = [x_min + i * dx for i in range(nx)]

    # CFL
    r = c * dt / dx
    if abs(r) > 1:
        raise ValueError(
            f"CFL condition violated: |c*dt/dx| = {abs(r):.4f} > 1")

    # Time steps
    nt = int(t_max / dt) + 1
    t = [i * dt for i in range(nt)]

    # Initialize
    u = []
    u_curr = [u0(xi) for xi in x]
    u.append(list(u_curr))

    # Time stepping
    for n in range(nt - 1):
        u_next = [0.0] * nx

        # Boundary conditions
        u_next[0] = boundary_conditions['left']
        u_next[-1] = boundary_conditions['right']

        # Lax-Wendroff: u_new = u - r/2*(u[i+1]-u[i-1]) + r²/2*(u[i+1]-2u[i]+u[i-1])
        for i in range(1, nx - 1):
            u_next[i] = (u_curr[i] - r * 0.5 *
                         (u_curr[i + 1] - u_curr[i - 1]) + r * r * 0.5 *
                         (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]))

        u_curr = u_next
        u.append(list(u_curr))

    return x, t, u


def main():
    """Example usage of 1D wave equation solvers."""
    import math

    print("=" * 70)
    print("1D Wave Equation: ∂²u/∂t² = c² ∂²u/∂x²")
    print("=" * 70)

    # Example 1: Gaussian pulse
    print("\n1. Wave equation with Gaussian pulse")
    print("   u(x,0) = exp(-100(x-0.5)²), ∂u/∂t(x,0) = 0")

    c = 1.0

    def u0_gauss(x):
        return math.exp(-100 * (x - 0.5)**2)

    def v0_zero(x):
        return 0

    bc = {'left': 0, 'right': 0}

    x, t, u = wave_explicit(c, 0, 1, 101, 0.5, 0.005, u0_gauss, v0_zero, bc)

    print(f"\n   CFL number: c*dt/dx = {c * 0.005 / 0.01:.2f}")
    print(f"   Time steps: {len(t)}")
    print(f"\n   t\t\tMax |u|\t\tPosition of max")
    print("   " + "-" * 50)

    for i in [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, -1]:
        max_u = max(abs(ui) for ui in u[i])
        pos_max = x[u[i].index(max(u[i], key=abs))]
        print(f"   {t[i]:.3f}\t{max_u:.6f}\t{pos_max:.3f}")

    # Example 2: Sine wave
    print("\n2. Standing wave: u(x,0) = sin(2πx)")

    def u0_sine(x):
        return math.sin(2 * math.pi * x)

    x, t, u_std = wave_explicit(c, 0, 1, 51, 2.0, 0.01, u0_sine, v0_zero, bc)
    x, t, u_lf = wave_lax_friedrichs(c, 0, 1, 51, 2.0, 0.01, u0_sine, v0_zero,
                                     bc)

    print(f"\n   Method\t\tMax |u| at t=2.0\tDissipation")
    print("   " + "-" * 50)
    max_std = max(abs(ui) for ui in u_std[-1])
    max_lf = max(abs(ui) for ui in u_lf[-1])
    print(f"   Standard\t{max_std:.6f}\t\t{(1-max_std)*100:.1f}%")
    print(f"   Lax-Friedrichs\t{max_lf:.6f}\t\t{(1-max_lf)*100:.1f}%")

    # Example 3: Advection - traveling pulse
    print("\n3. Advection equation: ∂u/∂t + c ∂u/∂x = 0")
    print("   Square pulse traveling to the right")

    def u0_square(x):
        return 1.0 if 0.3 <= x <= 0.5 else 0.0

    bc_adv = {'left': 0, 'right': 0}

    x, t, u_up = wave_upwind(1.0, 0, 1, 201, 0.3, 0.002, u0_square, bc_adv)
    x, t, u_lw = wave_lax_wendroff(1.0, 0, 1, 201, 0.3, 0.002, u0_square,
                                   bc_adv)

    print(f"\n   Time: t = {t[-1]:.2f}, pulse should be at x ≈ 0.6")
    print(f"\n   Method\t\tMax u\t\tPulse quality")
    print("   " + "-" * 50)
    max_up = max(u_up[-1])
    max_lw = max(u_lw[-1])
    print(
        f"   Upwind\t\t{max_up:.4f}\t\t{'Diffusive' if max_up < 0.8 else 'Good'}"
    )
    print(
        f"   Lax-Wendroff\t{max_lw:.4f}\t\t{'Excellent' if max_lw > 0.9 else 'Good'}"
    )

    # Example 4: CFL condition demonstration
    print("\n4. CFL Stability Condition: c*dt/dx ≤ 1")

    def u0_test(x):
        return math.sin(math.pi * x)

    print(f"\n   Testing different CFL numbers:")

    for cfl in [0.5, 0.9, 1.0]:
        dt_test = cfl * 0.02 / c  # dx = 0.02
        try:
            x, t, u_test = wave_explicit(c, 0, 1, 51, 0.1, dt_test, u0_test,
                                         v0_zero, bc)
            max_final = max(abs(ui) for ui in u_test[-1])
            stability = "Stable" if max_final < 2 else "Unstable"
            print(f"   CFL = {cfl:.1f}: {stability}, max|u| = {max_final:.6f}")
        except ValueError as e:
            print(f"   CFL = {cfl:.1f}: Rejected - {e}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("  - Standard explicit: Simple, needs CFL ≤ 1")
    print("  - Lax-Friedrichs: More stable but dissipative")
    print("  - Upwind: Good for advection, first-order accurate")
    print("  - Lax-Wendroff: Second-order, minimal dissipation")
    print("=" * 70)


if __name__ == "__main__":
    main()
