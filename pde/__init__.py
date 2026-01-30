"""
Partial Differential Equations (PDE) numerical methods.
"""

from .heat_1d import heat_explicit, heat_implicit, heat_crank_nicolson
from .poisson_2d import poisson_jacobi, poisson_gauss_seidel, poisson_sor
from .wave_1d import wave_explicit, wave_lax_friedrichs, wave_upwind, wave_lax_wendroff

__all__ = [
    "heat_explicit",
    "heat_implicit",
    "heat_crank_nicolson",
    "poisson_jacobi",
    "poisson_gauss_seidel",
    "poisson_sor",
    "wave_explicit",
    "wave_lax_friedrichs",
    "wave_upwind",
    "wave_lax_wendroff",
]
