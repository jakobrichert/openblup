"""OpenBLUP: Open-source REML and BLUP for plant and animal breeding.

The heavy lifting is done by the Rust extension module ``openblup._internal``;
this package re-exports it and adds a few conveniences (scipy sparse
matrices, pandas data frames).

Quick start::

    from openblup import MixedModel, Pedigree

    model = MixedModel()
    model.load_csv("field_trial.csv")
    model.set_response("yield")
    model.add_fixed("mu + rep")
    model.add_random("genotype")
    result = model.fit()
    print(result.summary())

    # Animal model with a pedigree relationship matrix
    ped = Pedigree.from_csv("pedigree.csv")
    model = MixedModel()
    model.load_csv("records.csv")
    model.set_response("weight")
    model.add_fixed("sex")
    model.add_random_pedigree("animal", ped)
    result = model.fit()
    print(result.variance_components())

    # Multi-environment trial: factor-analytic genotype-by-environment term
    model = MixedModel()
    model.load_csv("met_trial.csv")
    model.set_response("yield")
    model.add_fixed("mu + env")
    model.add_random_interaction("env", "genotype", outer_structure="fa1")
    result = model.fit()
    for p in result.variance_parameters():
        print(p["component"], p["name"], p["value"], p["se"])

    # Spatial analysis: AR1 x AR1 residual over the field grid
    model = MixedModel()
    model.load_csv("field_trial.csv")
    model.set_response("yield")
    model.add_fixed("mu + rep")
    model.add_random("genotype")
    model.set_residual_interaction("row", "col", "ar1", "ar1c")
    print(model.fit().summary())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import _internal
except ImportError as exc:  # pragma: no cover - only hit on a broken install
    raise ImportError(
        "OpenBLUP native extension not found. Build it with "
        "`pip install .` (or `maturin develop`), which requires a Rust toolchain."
    ) from exc

from ._internal import FitResult, Pedigree, __version__, compute_g_matrix, compute_inbreeding
from ._internal import MixedModel as _MixedModel
from ._internal import compute_a_inverse as _compute_a_inverse

SparseCSCTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]
"""(data, indices, indptr, shape) — the scipy CSC layout used at the Rust boundary."""


def _as_csc_tuple(matrix: Any) -> SparseCSCTuple:
    """Coerce a scipy sparse matrix, a dense 2-D array or a CSC tuple to a CSC tuple."""
    if isinstance(matrix, tuple) and len(matrix) == 4:
        data, indices, indptr, shape = matrix
        return (
            np.ascontiguousarray(data, dtype=np.float64),
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(indptr, dtype=np.int64),
            (int(shape[0]), int(shape[1])),
        )
    if hasattr(matrix, "tocsc"):
        m = matrix.tocsc()
        m.sum_duplicates()
        return (
            np.ascontiguousarray(m.data, dtype=np.float64),
            np.ascontiguousarray(m.indices, dtype=np.int64),
            np.ascontiguousarray(m.indptr, dtype=np.int64),
            (int(m.shape[0]), int(m.shape[1])),
        )
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("relationship matrix inverse must be 2-dimensional")
    # Dense -> CSC without requiring scipy.
    nrows, ncols = arr.shape
    data: List[float] = []
    indices: List[int] = []
    indptr = [0]
    for j in range(ncols):
        col = arr[:, j]
        nz = np.nonzero(col)[0]
        data.extend(col[nz].tolist())
        indices.extend(nz.tolist())
        indptr.append(len(data))
    return (
        np.asarray(data, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        (int(nrows), int(ncols)),
    )


def to_scipy_sparse(matrix: SparseCSCTuple):
    """Convert a (data, indices, indptr, shape) tuple to a ``scipy.sparse.csc_matrix``."""
    from scipy import sparse  # local import: scipy is optional

    data, indices, indptr, shape = matrix
    return sparse.csc_matrix((data, indices, indptr), shape=shape)


def to_dense(matrix: SparseCSCTuple) -> np.ndarray:
    """Convert a (data, indices, indptr, shape) tuple to a dense numpy array."""
    data, indices, indptr, shape = matrix
    out = np.zeros(shape, dtype=np.float64)
    for j in range(shape[1]):
        for k in range(indptr[j], indptr[j + 1]):
            out[indices[k], j] += data[k]
    return out


def compute_a_inverse(
    pedigree: Pedigree, inbreeding: bool = True, as_scipy: Optional[bool] = None
):
    """Compute the A-inverse of a pedigree.

    The pedigree is sorted in place if necessary; the row/column order of the
    result is ``pedigree.animal_ids()`` after the call. Pass those IDs as the
    ``levels`` of :meth:`MixedModel.add_random` when using the matrix directly
    (or simply use :meth:`MixedModel.add_random_pedigree`).

    Parameters
    ----------
    pedigree : Pedigree
    inbreeding : bool
        Use the Meuwissen & Luo (1992) inbreeding-aware rules (default True).
    as_scipy : bool or None
        Return a ``scipy.sparse.csc_matrix`` (True), the raw CSC tuple (False)
        or a scipy matrix when scipy is installed and the tuple otherwise (None).
    """
    result = _compute_a_inverse(pedigree, inbreeding)
    if as_scipy is False:
        return result
    try:
        return to_scipy_sparse(result)
    except ImportError:
        if as_scipy:
            raise
        return result


class MixedModel(_MixedModel):
    """Mixed model builder (see :mod:`openblup` for a quick start).

    Extends the native class with pandas and scipy conveniences.
    """

    def set_dataframe(self, df: Any) -> None:
        """Load data from a pandas ``DataFrame``.

        Numeric columns become float columns (missing values stay ``NaN`` and
        rows with a missing response are dropped at fit time); everything else
        becomes a factor.
        """
        columns: Dict[str, Any] = {}
        for name in df.columns:
            series = df[name]
            kind = getattr(series.dtype, "kind", "O")
            if kind in "biuf":
                columns[str(name)] = np.asarray(series, dtype=np.float64)
            else:
                columns[str(name)] = ["" if v is None else str(v) for v in series.tolist()]
        self.set_data(columns)

    def add_random(
        self,
        column: str,
        ginverse: Any = None,
        levels: Optional[Sequence[str]] = None,
        structure: Optional[str] = None,
    ) -> None:
        """Add a random term; ``ginverse`` may be a scipy sparse matrix, a dense
        array or a CSC tuple. See the native docstring for details."""
        if ginverse is not None:
            ginverse = _as_csc_tuple(ginverse)
        super().add_random(
            column, ginverse, None if levels is None else list(levels), structure
        )


__all__ = [
    "FitResult",
    "MixedModel",
    "Pedigree",
    "SparseCSCTuple",
    "__version__",
    "compute_a_inverse",
    "compute_g_matrix",
    "compute_inbreeding",
    "to_dense",
    "to_scipy_sparse",
]
