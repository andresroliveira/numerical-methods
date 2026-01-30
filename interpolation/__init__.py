"""
Interpolation methods.
"""

from .lagrange import lagrange
from .newton_divided_diff import newton_divided_diff
from .spline import linear_spline, cubic_spline_natural
from .vandermonde import vandermonde

__all__ = [
    "lagrange", "newton_divided_diff", "linear_spline", "cubic_spline_natural",
    "vandermonde"
]
