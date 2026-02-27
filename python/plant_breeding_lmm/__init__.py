"""OpenBLUP: Open-source REML and BLUP for plant and animal breeding."""

try:
    from openblup import (
        PyPedigree as Pedigree,
        PyMixedModel as MixedModel,
        PyFitResult as FitResult,
        compute_a_inverse,
        compute_g_matrix,
    )
except ImportError:
    raise ImportError(
        "OpenBLUP native extension not found. "
        "Install with: pip install -e . (requires maturin)"
    )

__all__ = [
    "Pedigree",
    "MixedModel",
    "FitResult",
    "compute_a_inverse",
    "compute_g_matrix",
]
