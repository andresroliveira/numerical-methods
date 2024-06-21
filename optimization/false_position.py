def false_position(f, a, b, tol=1e-6, max_iter=100):
    """
    False position method for finding a root of a function.

    Parameters
    ----------
    f : function
        The function for which to find a root.
    a : float
        The left endpoint of the initial interval.
    b : float
        The right endpoint of the initial interval.
    tol : float
        The convergence tolerance.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    x : float
        The estimated root.

    Raises
    ------
    ValueError
        If the function does not have opposite signs at the endpoints.

    """
    if f(a) * f(b) > 0:
        raise ValueError("The function must have opposite signs at the endpoints.")

    for _ in range(max_iter):
        x = a - f(a) * (b - a) / (f(b) - f(a))
        if abs(f(x)) < tol or (b - a) / 2 < tol:
            return x
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

    return x

def f(x):
    return x**2 - 2


def main():
    root = false_position(f, 0, 2)
    print(root)


if __name__ == "__main__":
    main()
