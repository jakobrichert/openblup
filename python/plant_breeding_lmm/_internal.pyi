"""Type stubs for the OpenBLUP native extension module."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

class Pedigree:
    """A pedigree representing parent-offspring relationships.

    Used for computing the additive relationship matrix inverse (A-inverse)
    needed for pedigree-based BLUP.
    """

    def __init__(self) -> None:
        """Create an empty pedigree."""
        ...

    @staticmethod
    def from_csv(path: str) -> "Pedigree":
        """Load a pedigree from a CSV file.

        The CSV must have columns: animal, sire, dam.
        Unknown parents are coded as "0", "", or "NA".
        """
        ...

    def add_animal(
        self,
        animal: str,
        sire: Optional[str] = None,
        dam: Optional[str] = None,
    ) -> None:
        """Add an animal to the pedigree."""
        ...

    def n_animals(self) -> int:
        """Return the number of animals in the pedigree."""
        ...

    def sort(self) -> None:
        """Sort the pedigree topologically (parents before offspring).

        This must be called before computing A-inverse.
        """
        ...

    def validate(self) -> None:
        """Validate the pedigree for consistency."""
        ...

    def is_sorted(self) -> bool:
        """Return whether the pedigree is topologically sorted."""
        ...

    def compute_a_inverse(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64], Tuple[int, int]]:
        """Compute A-inverse (Henderson's rules, no inbreeding).

        The pedigree must be sorted first (call .sort()).

        Returns
        -------
        tuple
            (data, indices, indptr, shape) for constructing a scipy.sparse.csc_matrix.
        """
        ...

    def compute_a_inverse_with_inbreeding(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64], Tuple[int, int]]:
        """Compute A-inverse with inbreeding (Meuwissen & Luo algorithm).

        The pedigree must be sorted first (call .sort()).

        Returns
        -------
        tuple
            (data, indices, indptr, shape) for constructing a scipy.sparse.csc_matrix.
        """
        ...

    def compute_inbreeding(self) -> npt.NDArray[np.float64]:
        """Compute inbreeding coefficients for all animals.

        Returns
        -------
        numpy.ndarray
            A 1D array of inbreeding coefficients.
        """
        ...

    def animal_ids(self) -> List[str]:
        """Get the list of animal IDs in pedigree order."""
        ...


SparseCSCTuple = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    Tuple[int, int],
]
"""Type alias for a scipy-compatible CSC sparse matrix tuple:
(data, indices, indptr, shape)."""


class MixedModel:
    """A mixed model builder that accumulates data, fixed effects, and random effects,
    then fits the model via REML."""

    def __init__(self) -> None:
        """Create a new mixed model builder."""
        ...

    def set_data(
        self,
        columns: Dict[str, Union[List[float], List[int], List[str], npt.NDArray]],
    ) -> None:
        """Set data from a dictionary of column name to values.

        Values can be:
          - list of floats or numpy array of floats -> Float column
          - list of strings -> Factor (categorical) column
          - list of ints or numpy array of ints -> Integer column
        """
        ...

    def load_csv(self, path: str) -> None:
        """Load data from a CSV file.

        Numeric columns are auto-detected as Float; others become Factor.
        Use ``as_factor(column)`` to convert numeric columns to categorical.
        """
        ...

    def as_factor(self, column: str) -> None:
        """Convert a column to a Factor (categorical) column."""
        ...

    def set_response(self, column: str) -> None:
        """Set the response (dependent) variable."""
        ...

    def add_fixed(self, formula: str) -> None:
        """Set the fixed effects formula.

        Examples: "mu", "mu + rep", "mu + rep + block".
        "mu" or "intercept" or "1" adds an intercept.
        """
        ...

    def add_random(
        self,
        column: str,
        ginverse: Optional[SparseCSCTuple] = None,
    ) -> None:
        """Add a random effect term.

        Parameters
        ----------
        column : str
            The factor column to use as the grouping variable.
        ginverse : tuple or None
            Optional relationship matrix inverse as (data, indices, indptr, shape)
            from a scipy.sparse.csc_matrix. If None, an identity matrix is used.
        """
        ...

    def set_max_iterations(self, n: int) -> None:
        """Set maximum REML iterations (default: 50)."""
        ...

    def set_convergence_tol(self, tol: float) -> None:
        """Set convergence tolerance (default: 1e-6)."""
        ...

    def fit(self) -> "FitResult":
        """Fit the model using REML.

        Returns
        -------
        FitResult
            The fitted model results.
        """
        ...


class FitResult:
    """The result of fitting a mixed model via REML."""

    converged: bool
    """Whether the REML algorithm converged."""

    n_iterations: int
    """Number of iterations performed."""

    n_obs: int
    """Number of observations."""

    n_fixed_params: int
    """Number of fixed effect parameters."""

    n_variance_params: int
    """Number of variance parameters."""

    def summary(self) -> str:
        """Print a formatted summary of the model fit."""
        ...

    def variance_components(self) -> Dict[str, float]:
        """Return the variance components as a dict of {name: sigma2_value}."""
        ...

    def fixed_effects(self) -> List[Tuple[str, str, float, float]]:
        """Return the fixed effects as a list of (term, level, estimate, se) tuples."""
        ...

    def random_effects(self) -> Dict[str, npt.NDArray[np.float64]]:
        """Return the random effects as a dict of {term_name: numpy_array}."""
        ...

    def random_effect_levels(self) -> Dict[str, List[str]]:
        """Return the random effect level names for each term."""
        ...

    def residuals(self) -> npt.NDArray[np.float64]:
        """Return the residuals as a numpy array."""
        ...

    def log_likelihood(self) -> float:
        """Return the restricted log-likelihood."""
        ...

    def aic(self) -> float:
        """Return the AIC."""
        ...

    def bic(self) -> float:
        """Return the BIC."""
        ...

    def iteration_history(self) -> List[Dict[str, object]]:
        """Return the iteration history as a list of dicts."""
        ...


def compute_a_inverse(
    ped: Pedigree,
) -> SparseCSCTuple:
    """Compute the A-inverse matrix from a pedigree.

    This is a convenience function that sorts the pedigree and computes A-inverse
    in a single call.

    Parameters
    ----------
    ped : Pedigree
        The pedigree to compute A-inverse for.

    Returns
    -------
    tuple
        (data, indices, indptr, shape) for scipy.sparse.csc_matrix.
    """
    ...


def compute_g_matrix(
    markers: npt.NDArray[np.float64],
    allele_freqs: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """Compute the genomic relationship matrix (G) using VanRaden Method 1.

    Parameters
    ----------
    markers : numpy.ndarray
        An (n_individuals x n_markers) matrix with 0/1/2 coding.
    allele_freqs : numpy.ndarray or None
        Optional allele frequencies. If None, estimated from marker data.

    Returns
    -------
    numpy.ndarray
        The G-matrix of dimension (n_individuals x n_individuals).
    """
    ...
