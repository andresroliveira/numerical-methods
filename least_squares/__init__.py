"""
Least squares fitting methods.
"""

from .linear_fit import linear_fit
from .polynomial_fit import polynomial_fit
from .fourier_fit import fourier_fit
from .exponential_fit import exponential_fit
from .general_fit import general_fit

__all__ = [
    "linear_fit",
    "polynomial_fit",
    "fourier_fit",
    "exponential_fit",
    "general_fit",
]
