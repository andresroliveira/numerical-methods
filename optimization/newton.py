def newton(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding a root of a function.

    Parameters
    ----------
    f : function
        The function for which to find a root.
    df : function
        The derivative of the function.
    x0 : float
        The initial guess for the root.
    tol : float
        The convergence tolerance.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    x : float
        The estimated root.

    """
    for _ in range(max_iter):
        x = x0 - f(x0) / df(x0)
        if abs(x - x0) < tol:
            return x
        x0 = x

    return x


def f(x):
    return x**2 - 2


def df(x):
    return 2 * x


def main():
    root = newton(f, df, 1)
    print(root)


if __name__ == "__main__":
    main()
