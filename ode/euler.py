"""
    Given x' = f(x), find x(t) for t in [0, T] with x(0) = x0.
"""


def euler(f, x0, T, n):
    """
    Solve x' = f(x), x(0) = x0, with the Euler method.

    Parameters
    ----------
    f : function
        The function f(x) such that x' = f(x).
    x0 : float
        The initial condition x(0) = x0.
    T : float
        The final time.
    n : int
        The number of steps to take.

    Returns
    -------
    list of float
        The solution x(t) at each time t.
    """
    x = x0
    dt = T / n
    result = [x0]
    for _ in range(n):
        x += dt * f(x)
        result.append(x)

    return result


def f(x):
    return (1 - x) * x


def main():
    x0 = 0.1
    T = 5
    n = 100
    result = euler(f, x0, T, n)
    print(result)


if __name__ == "__main__":
    main()
