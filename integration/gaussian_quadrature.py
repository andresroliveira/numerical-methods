def gaussian_quadrature_2(f, a, b):
    """
    Compute the integral of f from a to b using 2-point Gaussian quadrature.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.

    Returns
    -------
    float
        The estimated integral value.

    """
    # 2-point Gauss-Legendre quadrature nodes and weights
    # on interval [-1, 1]
    nodes = [-0.5773502691896257, 0.5773502691896257]
    weights = [1.0, 1.0]

    # Transform from [-1, 1] to [a, b]
    # x = ((b-a)*t + (b+a)) / 2
    mid = (b + a) / 2
    half = (b - a) / 2

    result = 0.0
    for i in range(2):
        x = mid + half * nodes[i]
        result += weights[i] * f(x)

    return half * result


def gaussian_quadrature_3(f, a, b):
    """
    Compute the integral of f from a to b using 3-point Gaussian quadrature.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.

    Returns
    -------
    float
        The estimated integral value.

    """
    # 3-point Gauss-Legendre quadrature nodes and weights
    nodes = [-0.7745966692414834, 0.0, 0.7745966692414834]
    weights = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]

    mid = (b + a) / 2
    half = (b - a) / 2

    result = 0.0
    for i in range(3):
        x = mid + half * nodes[i]
        result += weights[i] * f(x)

    return half * result


def gaussian_quadrature_4(f, a, b):
    """
    Compute the integral of f from a to b using 4-point Gaussian quadrature.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.

    Returns
    -------
    float
        The estimated integral value.

    """
    # 4-point Gauss-Legendre quadrature nodes and weights
    nodes = [
        -0.8611363115940526,
        -0.3399810435848563,
        0.3399810435848563,
        0.8611363115940526,
    ]
    weights = [
        0.3478548451374538,
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374538,
    ]

    mid = (b + a) / 2
    half = (b - a) / 2

    result = 0.0
    for i in range(4):
        x = mid + half * nodes[i]
        result += weights[i] * f(x)

    return half * result


def gaussian_quadrature(f, a, b, n=3):
    """
    Compute the integral of f from a to b using n-point Gaussian quadrature.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    n : int, optional
        The number of quadrature points (2, 3, or 4). Default is 3.

    Returns
    -------
    float
        The estimated integral value.

    Raises
    ------
    ValueError
        If n is not 2, 3, or 4.

    """
    if n == 2:
        return gaussian_quadrature_2(f, a, b)
    elif n == 3:
        return gaussian_quadrature_3(f, a, b)
    elif n == 4:
        return gaussian_quadrature_4(f, a, b)
    else:
        raise ValueError("n must be 2, 3, or 4")


def main():
    # Example: integrate f(x) = x^2 from 0 to 1
    # Exact answer: 1/3 = 0.333333...
    import math

    def f(x):
        return x**2

    print("Integrating f(x) = x^2 from 0 to 1")
    print(f"Exact value: {1/3:.10f}")
    print()

    for n in [2, 3, 4]:
        result = gaussian_quadrature(f, 0, 1, n)
        error = abs(result - 1 / 3)
        print(f"{n}-point Gauss: {result:.10f}  (error: {error:.2e})")

    print()

    # Example 2: integrate sin(x) from 0 to pi
    # Exact answer: 2
    def g(x):
        return math.sin(x)

    print("Integrating sin(x) from 0 to π")
    print(f"Exact value: {2.0:.10f}")
    print()

    for n in [2, 3, 4]:
        result = gaussian_quadrature(g, 0, math.pi, n)
        error = abs(result - 2.0)
        print(f"{n}-point Gauss: {result:.10f}  (error: {error:.2e})")


if __name__ == "__main__":
    main()
