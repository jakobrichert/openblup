//! Python bindings for OpenBLUP (PyO3).
//!
//! The extension module is published as `openblup._internal`; the pure-Python
//! package in `python/openblup/__init__.py` re-exports and lightly wraps it
//! (scipy/pandas conveniences).

use std::collections::HashMap;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use plant_breeding_lmm_core as core;
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::diagnostics::{wald_tests, ResidualDiagnostics, WaldTest};
use plant_breeding_lmm_core::genetics::Pedigree as CorePedigree;
use plant_breeding_lmm_core::lmm::{reliability_from_se, FitResult as CoreFitResult};
use plant_breeding_lmm_core::model::{MixedModel, MixedModelBuilder};
use plant_breeding_lmm_core::variance::{Identity, StructureSpec};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a core library error into a Python ValueError.
fn to_pyerr(e: core::LmmError) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

/// Convert a sparse CSC matrix into the (data, indices, indptr, shape) tuple
/// that Python/scipy expects for constructing a `csc_matrix`.
fn sparse_to_scipy_csc(py: Python<'_>, mat: &sprs::CsMat<f64>) -> PyResult<PyObject> {
    let csc = if mat.is_csc() {
        mat.clone()
    } else {
        mat.to_csc()
    };

    let data_vec: Vec<f64> = csc.data().to_vec();
    let indices_vec: Vec<i64> = csc.indices().iter().map(|&i| i as i64).collect();
    let indptr_raw = csc.indptr();
    let indptr_slice = indptr_raw.as_slice().unwrap();
    let indptr_vec: Vec<i64> = indptr_slice.iter().map(|&i| i as i64).collect();
    let shape = (csc.rows(), csc.cols());

    let data = PyArray1::from_vec(py, data_vec);
    let indices = PyArray1::from_vec(py, indices_vec);
    let indptr = PyArray1::from_vec(py, indptr_vec);

    Ok((data, indices, indptr, shape).into_pyobject(py)?.into())
}

/// Parse a (data, indices, indptr, shape) tuple into a sparse CSC matrix.
fn scipy_csc_to_sparse(py: Python<'_>, obj: &PyObject) -> PyResult<sprs::CsMat<f64>> {
    let (data_arr, indices_arr, indptr_arr, shape) = obj.extract::<(
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<i64>,
        PyReadonlyArray1<i64>,
        (usize, usize),
    )>(py)?;

    let data: Vec<f64> = data_arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read ginverse data: {}", e)))?
        .to_vec();
    let indices: Vec<usize> = indices_arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read ginverse indices: {}", e)))?
        .iter()
        .map(|&i| i as usize)
        .collect();
    let indptr: Vec<usize> = indptr_arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Failed to read ginverse indptr: {}", e)))?
        .iter()
        .map(|&i| i as usize)
        .collect();

    if shape.0 != shape.1 {
        return Err(PyValueError::new_err(format!(
            "ginverse must be square, got {}x{}",
            shape.0, shape.1
        )));
    }
    if indptr.len() != shape.1 + 1 {
        return Err(PyValueError::new_err(format!(
            "ginverse indptr has length {} but {} columns require {}",
            indptr.len(),
            shape.1,
            shape.1 + 1
        )));
    }

    sprs::CsMat::try_new_csc((shape.0, shape.1), indptr, indices, data)
        .map_err(|e| PyValueError::new_err(format!("Invalid CSC matrix: {:?}", e)))
}

// ---------------------------------------------------------------------------
// Pedigree
// ---------------------------------------------------------------------------

/// A pedigree representing parent-offspring relationships.
///
/// Used for computing the additive relationship matrix inverse (A-inverse)
/// needed for pedigree-based BLUP.
#[pyclass(name = "Pedigree")]
#[derive(Clone)]
struct PyPedigree {
    inner: CorePedigree,
}

#[pymethods]
impl PyPedigree {
    /// Create an empty pedigree.
    #[new]
    fn new() -> Self {
        PyPedigree {
            inner: CorePedigree::new(),
        }
    }

    /// Load a pedigree from a CSV file.
    ///
    /// The CSV must have columns: animal, sire, dam (case-insensitive).
    /// Unknown parents are coded as "0", "", or "NA".
    #[staticmethod]
    fn from_csv(path: &str) -> PyResult<Self> {
        let ped = CorePedigree::from_csv(path).map_err(to_pyerr)?;
        Ok(PyPedigree { inner: ped })
    }

    /// Build a pedigree from (animal, sire, dam) triples. Use None, "0", ""
    /// or "NA" for unknown parents.
    #[staticmethod]
    fn from_triples(triples: Vec<(String, Option<String>, Option<String>)>) -> PyResult<Self> {
        let unknown = |s: Option<String>| {
            s.filter(|v| !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("na")))
        };
        let triples: Vec<(String, Option<String>, Option<String>)> = triples
            .into_iter()
            .map(|(a, s, d)| (a, unknown(s), unknown(d)))
            .collect();
        let ped = CorePedigree::from_triples(&triples).map_err(to_pyerr)?;
        Ok(PyPedigree { inner: ped })
    }

    /// Add an animal to the pedigree.
    ///
    /// Parameters
    /// ----------
    /// animal : str
    ///     The animal identifier.
    /// sire : str or None
    ///     The sire identifier, or None if unknown.
    /// dam : str or None
    ///     The dam identifier, or None if unknown.
    #[pyo3(signature = (animal, sire=None, dam=None))]
    fn add_animal(&mut self, animal: &str, sire: Option<&str>, dam: Option<&str>) -> PyResult<()> {
        self.inner.add_animal(animal, sire, dam).map_err(to_pyerr)
    }

    /// Return the number of animals in the pedigree.
    fn n_animals(&self) -> usize {
        self.inner.n_animals()
    }

    fn __len__(&self) -> usize {
        self.inner.n_animals()
    }

    /// Sort the pedigree topologically (parents before offspring).
    ///
    /// This must be called before computing A-inverse.
    fn sort(&mut self) -> PyResult<()> {
        self.inner.sort_pedigree().map_err(to_pyerr)
    }

    /// Validate the pedigree for consistency.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_pyerr)
    }

    /// Return whether the pedigree is topologically sorted.
    fn is_sorted(&self) -> bool {
        self.inner.is_sorted()
    }

    /// Compute A-inverse (Henderson's rules, ignoring inbreeding).
    ///
    /// The pedigree must be sorted first (call .sort()).
    ///
    /// Returns
    /// -------
    /// tuple
    ///     (data, indices, indptr, shape) for constructing a scipy.sparse.csc_matrix.
    fn compute_a_inverse(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ainv = core::genetics::compute_a_inverse(&self.inner).map_err(to_pyerr)?;
        sparse_to_scipy_csc(py, &ainv)
    }

    /// Compute A-inverse with inbreeding (Meuwissen & Luo algorithm).
    ///
    /// The pedigree must be sorted first (call .sort()).
    ///
    /// Returns
    /// -------
    /// tuple
    ///     (data, indices, indptr, shape) for constructing a scipy.sparse.csc_matrix.
    fn compute_a_inverse_with_inbreeding(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ainv =
            core::genetics::compute_a_inverse_with_inbreeding(&self.inner).map_err(to_pyerr)?;
        sparse_to_scipy_csc(py, &ainv)
    }

    /// Compute inbreeding coefficients for all animals (pedigree order).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A 1D array of inbreeding coefficients.
    fn compute_inbreeding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = core::genetics::compute_inbreeding(&self.inner).map_err(to_pyerr)?;
        Ok(PyArray1::from_vec(py, f))
    }

    /// Get the list of animal IDs in pedigree order. After `sort()` this is
    /// the row/column order of the A-inverse.
    fn animal_ids(&self) -> Vec<String> {
        (0..self.inner.n_animals())
            .map(|i| self.inner.animal_id(i).to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Pedigree(n_animals={}, sorted={})",
            self.inner.n_animals(),
            self.inner.is_sorted()
        )
    }
}

// ---------------------------------------------------------------------------
// MixedModel
// ---------------------------------------------------------------------------

/// Internal representation of a random term specification.
struct RandomTermPy {
    column: String,
    ginv: Option<sprs::CsMat<f64>>,
    levels: Option<Vec<String>>,
    pedigree: Option<CorePedigree>,
    /// Variance structure spec (None = scaled identity).
    structure: Option<StructureSpec>,
    /// Interaction: inner factor column and its structure spec.
    interaction: Option<(String, Option<StructureSpec>)>,
}

/// Residual specification.
enum ResidualPy {
    Structure(StructureSpec),
    Grid {
        row: String,
        row_structure: StructureSpec,
        col: String,
        col_structure: StructureSpec,
    },
}

fn parse_structure(spec: &str) -> PyResult<StructureSpec> {
    StructureSpec::parse(spec).map_err(to_pyerr)
}

/// A mixed model builder that accumulates data, fixed effects, and random effects,
/// then fits the model via REML.
#[pyclass(name = "MixedModel", subclass)]
struct PyMixedModel {
    df: Option<DataFrame>,
    response: Option<String>,
    fixed_formula: Option<String>,
    random_terms: Vec<RandomTermPy>,
    residual: Option<ResidualPy>,
    max_iter: usize,
    convergence_tol: f64,
    algorithm: String,
    drop_missing_response: bool,
}

#[pymethods]
impl PyMixedModel {
    /// Create a new mixed model builder.
    #[new]
    fn new() -> Self {
        PyMixedModel {
            df: None,
            response: None,
            fixed_formula: None,
            random_terms: Vec::new(),
            residual: None,
            max_iter: 50,
            convergence_tol: 1e-6,
            algorithm: "ai".to_string(),
            drop_missing_response: true,
        }
    }

    /// Set data from a dictionary of column name -> values.
    ///
    /// Values can be:
    ///   - list of floats or numpy array of floats -> Float column
    ///   - list of strings -> Factor column
    ///   - list of ints or numpy array of ints -> Integer column (use as_factor to convert)
    ///
    /// Parameters
    /// ----------
    /// columns : dict
    ///     Mapping of column names to column data.
    fn set_data(&mut self, py: Python<'_>, columns: HashMap<String, PyObject>) -> PyResult<()> {
        let mut df = DataFrame::new();

        for (name, obj) in &columns {
            // Try to extract as numpy f64 array first
            if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>(py) {
                let data: Vec<f64> = arr
                    .as_slice()
                    .map_err(|e| {
                        PyValueError::new_err(format!("Failed to read array for '{}': {}", name, e))
                    })?
                    .to_vec();
                df.add_float_column(name, data).map_err(to_pyerr)?;
                continue;
            }

            // Try numpy i64 array
            if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>(py) {
                let data: Vec<i64> = arr
                    .as_slice()
                    .map_err(|e| {
                        PyValueError::new_err(format!("Failed to read array for '{}': {}", name, e))
                    })?
                    .to_vec();
                df.add_integer_column(name, data).map_err(to_pyerr)?;
                continue;
            }

            // Try list of ints before floats so 1 stays an integer code.
            if let Ok(vals) = obj.extract::<Vec<i64>>(py) {
                df.add_integer_column(name, vals).map_err(to_pyerr)?;
                continue;
            }

            // Try list of floats
            if let Ok(vals) = obj.extract::<Vec<f64>>(py) {
                df.add_float_column(name, vals).map_err(to_pyerr)?;
                continue;
            }

            // Try list of strings -> Factor
            if let Ok(vals) = obj.extract::<Vec<String>>(py) {
                let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
                df.add_factor_column(name, &refs).map_err(to_pyerr)?;
                continue;
            }

            return Err(PyValueError::new_err(format!(
                "Column '{}': unsupported type. Expected list of float, int, or str, \
                 or a numpy array.",
                name
            )));
        }

        self.df = Some(df);
        Ok(())
    }

    /// Load data from a CSV file.
    ///
    /// Numeric columns are auto-detected as Float (with NA/empty fields stored
    /// as NaN); others become Factor. Use `as_factor(column)` to convert
    /// numeric columns to categorical.
    fn load_csv(&mut self, path: &str) -> PyResult<()> {
        let df = DataFrame::from_csv(path).map_err(to_pyerr)?;
        self.df = Some(df);
        Ok(())
    }

    /// Convert a column to a Factor (categorical) column.
    ///
    /// This is useful when numeric-coded columns (like block numbers read from
    /// CSV) should be treated as categorical rather than continuous.
    fn as_factor(&mut self, column: &str) -> PyResult<()> {
        let df = self.df.as_mut().ok_or_else(|| {
            PyValueError::new_err("No data loaded. Call set_data() or load_csv() first.")
        })?;
        df.as_factor(column).map_err(to_pyerr)
    }

    /// Drop rows where the given numeric column is missing (NaN).
    ///
    /// Rows with a missing response are dropped automatically at fit time, so
    /// this is only needed for other columns (e.g. covariates).
    fn drop_missing(&mut self, column: &str) -> PyResult<usize> {
        let df = self.df.as_mut().ok_or_else(|| {
            PyValueError::new_err("No data loaded. Call set_data() or load_csv() first.")
        })?;
        let before = df.nrows();
        *df = df.drop_missing(column).map_err(to_pyerr)?;
        Ok(before - df.nrows())
    }

    /// Number of rows currently loaded.
    fn n_rows(&self) -> usize {
        self.df.as_ref().map(|d| d.nrows()).unwrap_or(0)
    }

    /// Column names currently loaded.
    fn columns(&self) -> Vec<String> {
        self.df
            .as_ref()
            .map(|d| d.column_names().iter().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }

    /// Set the response (dependent) variable.
    fn set_response(&mut self, column: &str) -> PyResult<()> {
        self.response = Some(column.to_string());
        Ok(())
    }

    /// Set the fixed effects formula.
    ///
    /// Examples: "mu", "mu + rep", "mu + rep + block"
    /// "mu" or "intercept" or "1" adds an intercept. Factors are coded with
    /// treatment contrasts (first level is the reference) when an intercept
    /// is present.
    fn add_fixed(&mut self, formula: &str) -> PyResult<()> {
        self.fixed_formula = Some(formula.to_string());
        Ok(())
    }

    /// Add a random effect term.
    ///
    /// Parameters
    /// ----------
    /// column : str
    ///     The factor column to use as the grouping variable. Numeric columns
    ///     are converted to factors automatically.
    /// ginverse : tuple or None
    ///     Optional relationship matrix inverse as (data, indices, indptr, shape)
    ///     from a scipy.sparse.csc_matrix. If None, an identity matrix is used.
    /// levels : list of str or None
    ///     Level order matching the rows/columns of `ginverse`. Required when
    ///     the matrix has its own ordering (e.g. `Pedigree.animal_ids()`);
    ///     levels without observations are allowed. If None, the levels are
    ///     the distinct values of the column in order of first appearance.
    /// structure : str or None
    ///     Variance structure: "idv" (default), "ar1", "ar1(rho)", "ar1c",
    ///     "diag", "us", "fa1", "fa2", ... A non-identity structure cannot be
    ///     combined with `ginverse`/`levels` (use add_random_interaction).
    #[pyo3(signature = (column, ginverse=None, levels=None, structure=None))]
    fn add_random(
        &mut self,
        py: Python<'_>,
        column: &str,
        ginverse: Option<PyObject>,
        levels: Option<Vec<String>>,
        structure: Option<&str>,
    ) -> PyResult<()> {
        let ginv = match ginverse {
            Some(obj) => Some(scipy_csc_to_sparse(py, &obj)?),
            None => None,
        };
        let structure = match structure {
            Some(s) => {
                let spec = parse_structure(s)?;
                if !spec.is_identity() && (ginv.is_some() || levels.is_some()) {
                    return Err(PyValueError::new_err(
                        "A relationship matrix can only be combined with the identity structure; \
                         use add_random_interaction(..., pedigree=...) for structured genetic effects",
                    ));
                }
                Some(spec)
            }
            None => None,
        };

        self.random_terms.push(RandomTermPy {
            column: column.to_string(),
            ginv,
            levels,
            pedigree: None,
            structure,
            interaction: None,
        });
        Ok(())
    }

    /// Add an interaction random term `outer:inner` with a separable
    /// covariance `Sigma_outer ⊗ Sigma_inner`.
    ///
    /// Parameters
    /// ----------
    /// outer : str
    ///     Outer factor column (e.g. environment), with structure
    ///     `outer_structure` ("diag" by default; "fa1", "fa2", "us", "ar1", ...).
    /// inner : str
    ///     Inner factor column (e.g. genotype or animal).
    /// inner_structure : str or None
    ///     Structure of the inner factor; None means independent levels.
    ///     Use a correlation-only structure ("ar1c") so the scale stays
    ///     identifiable.
    /// pedigree : Pedigree or None
    ///     If given, the inner factor uses the pedigree relationship matrix
    ///     (all pedigree animals become levels).
    #[pyo3(signature = (outer, inner, outer_structure="diag", inner_structure=None, pedigree=None))]
    fn add_random_interaction(
        &mut self,
        outer: &str,
        inner: &str,
        outer_structure: &str,
        inner_structure: Option<&str>,
        pedigree: Option<&PyPedigree>,
    ) -> PyResult<()> {
        let outer_spec = parse_structure(outer_structure)?;
        let inner_spec = match inner_structure {
            Some(s) => Some(parse_structure(s)?),
            None => None,
        };
        let ped = match pedigree {
            Some(p) => {
                let mut ped = p.inner.clone();
                if !ped.is_sorted() {
                    ped.sort_pedigree().map_err(to_pyerr)?;
                }
                Some(ped)
            }
            None => None,
        };
        self.random_terms.push(RandomTermPy {
            column: outer.to_string(),
            ginv: None,
            levels: None,
            pedigree: ped,
            structure: Some(outer_spec),
            interaction: Some((inner.to_string(), inner_spec)),
        });
        Ok(())
    }

    /// Set the residual variance structure applied to the observations in
    /// data order ("ar1", "ar1(rho)", ...). Default: IID.
    fn set_residual(&mut self, structure: &str) -> PyResult<()> {
        self.residual = Some(ResidualPy::Structure(parse_structure(structure)?));
        Ok(())
    }

    /// Set a separable residual `Sigma_row ⊗ Sigma_col` over the grid defined
    /// by two factor columns (e.g. field rows and columns); every observation
    /// must occupy a distinct cell, missing plots are allowed.
    #[pyo3(signature = (row, col, row_structure="ar1", col_structure="ar1c"))]
    fn set_residual_interaction(
        &mut self,
        row: &str,
        col: &str,
        row_structure: &str,
        col_structure: &str,
    ) -> PyResult<()> {
        self.residual = Some(ResidualPy::Grid {
            row: row.to_string(),
            row_structure: parse_structure(row_structure)?,
            col: col.to_string(),
            col_structure: parse_structure(col_structure)?,
        });
        Ok(())
    }

    /// Add a pedigree-based random effect (animal model).
    ///
    /// The term's levels are all animals of the pedigree (sorted order) and
    /// the A-inverse with inbreeding is used as the relationship matrix
    /// inverse. Every value in `column` must be an animal of the pedigree.
    fn add_random_pedigree(&mut self, column: &str, pedigree: &PyPedigree) -> PyResult<()> {
        let mut ped = pedigree.inner.clone();
        if !ped.is_sorted() {
            ped.sort_pedigree().map_err(to_pyerr)?;
        }
        self.random_terms.push(RandomTermPy {
            column: column.to_string(),
            ginv: None,
            levels: None,
            pedigree: Some(ped),
            structure: None,
            interaction: None,
        });
        Ok(())
    }

    /// Set maximum REML iterations (default: 50).
    fn set_max_iterations(&mut self, n: usize) {
        self.max_iter = n;
    }

    /// Set convergence tolerance (default: 1e-6).
    fn set_convergence_tol(&mut self, tol: f64) {
        self.convergence_tol = tol;
    }

    /// Choose the REML algorithm: "ai" (default, Average Information) or "em".
    fn set_algorithm(&mut self, algorithm: &str) -> PyResult<()> {
        match algorithm.to_lowercase().as_str() {
            "ai" | "ai-reml" => self.algorithm = "ai".into(),
            "em" | "em-reml" => self.algorithm = "em".into(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown algorithm '{}'. Use 'ai' or 'em'.",
                    other
                )))
            }
        }
        Ok(())
    }

    /// Whether rows with a missing response are dropped at fit time (default True).
    fn set_drop_missing_response(&mut self, drop: bool) {
        self.drop_missing_response = drop;
    }

    /// Fit the model using REML.
    ///
    /// Returns
    /// -------
    /// FitResult
    ///     The fitted model results.
    fn fit(&mut self) -> PyResult<PyFitResult> {
        let mut model = self.build_model()?;
        let result = match self.algorithm.as_str() {
            "em" => model.fit_em_reml(),
            _ => model.fit_reml(),
        }
        .map_err(to_pyerr)?;
        let diagnostics = result.residual_diagnostics(&model);
        Ok(PyFitResult {
            inner: result,
            diagnostics,
        })
    }

    /// k-fold cross-validation of prediction accuracy for a model with a
    /// single random term (genomic/pedigree prediction). Returns a dict with
    /// overall accuracy, msep, bias, mae and per-fold results.
    #[pyo3(signature = (n_folds=5, seed=0))]
    fn cross_validate<'py>(
        &mut self,
        py: Python<'py>,
        n_folds: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let model = self.build_model()?;
        let cv = model.cross_validate(n_folds, seed).map_err(to_pyerr)?;
        let d = PyDict::new(py);
        d.set_item("n_folds", cv.folds.len())?;
        d.set_item("accuracy", cv.accuracy)?;
        d.set_item("msep", cv.msep)?;
        d.set_item("bias", cv.bias)?;
        d.set_item("mae", cv.mae)?;
        let folds: Vec<Bound<'py, PyDict>> = cv
            .folds
            .iter()
            .map(|f| {
                let fd = PyDict::new(py);
                fd.set_item("fold", f.fold)?;
                fd.set_item("n_validation", f.validation_indices.len())?;
                fd.set_item("accuracy", f.accuracy)?;
                fd.set_item("msep", f.msep)?;
                Ok(fd)
            })
            .collect::<PyResult<_>>()?;
        d.set_item("folds", folds)?;
        Ok(d)
    }
}

impl PyMixedModel {
    /// Build the core model from the accumulated specification.
    fn build_model(&self) -> PyResult<MixedModel> {
        let mut df = self.df.clone().ok_or_else(|| {
            PyValueError::new_err("No data loaded. Call set_data() or load_csv() first.")
        })?;

        let response = self.response.clone().ok_or_else(|| {
            PyValueError::new_err("No response variable set. Call set_response() first.")
        })?;

        if self.drop_missing_response && df.has_missing(&response).map_err(to_pyerr)? {
            df = df.drop_missing(&response).map_err(to_pyerr)?;
        }
        let mut factor_columns: Vec<&str> = Vec::new();
        for rt in &self.random_terms {
            factor_columns.push(&rt.column);
            if let Some((inner, _)) = &rt.interaction {
                factor_columns.push(inner);
            }
        }
        if let Some(ResidualPy::Grid { row, col, .. }) = &self.residual {
            factor_columns.push(row);
            factor_columns.push(col);
        }
        for col in factor_columns {
            if df.get_factor(col).is_err() {
                df.as_factor(col).map_err(|e| {
                    PyValueError::new_err(format!(
                        "Term column '{}' must be a categorical column: {}",
                        col, e
                    ))
                })?;
            }
        }

        let mut builder = MixedModelBuilder::new()
            .data(&df)
            .response(&response)
            .max_iterations(self.max_iter)
            .convergence(self.convergence_tol);

        if let Some(ref formula) = self.fixed_formula {
            builder = builder.fixed(formula);
        }

        for rt in &self.random_terms {
            builder = match (&rt.interaction, &rt.pedigree, &rt.levels, &rt.structure) {
                (Some((inner, inner_spec)), ped, _, spec) => builder.random_interaction_spec(
                    &rt.column,
                    spec.clone()
                        .unwrap_or(StructureSpec::Diagonal { sigma2: 1.0 }),
                    inner,
                    inner_spec.clone(),
                    ped.as_ref(),
                ),
                (None, Some(ped), _, _) => {
                    builder.random_pedigree(&rt.column, Identity::new(1.0), ped)
                }
                (None, None, Some(levels), _) => builder.random_with_levels(
                    &rt.column,
                    Identity::new(1.0),
                    rt.ginv.clone(),
                    levels.clone(),
                ),
                (None, None, None, Some(spec)) if !spec.is_identity() => {
                    builder.random_spec(&rt.column, spec.clone(), None)
                }
                (None, None, None, _) => {
                    builder.random(&rt.column, Identity::new(1.0), rt.ginv.clone())
                }
            };
        }

        match &self.residual {
            Some(ResidualPy::Structure(spec)) => {
                builder = builder.residual_spec(spec.clone());
            }
            Some(ResidualPy::Grid {
                row,
                row_structure,
                col,
                col_structure,
            }) => {
                builder = builder.residual_interaction_spec(
                    row,
                    row_structure.clone(),
                    col,
                    col_structure.clone(),
                );
            }
            None => {}
        }

        builder.build().map_err(to_pyerr)
    }
}

// ---------------------------------------------------------------------------
// FitResult
// ---------------------------------------------------------------------------

/// The result of fitting a mixed model via REML.
#[pyclass(name = "FitResult")]
struct PyFitResult {
    inner: CoreFitResult,
    diagnostics: Option<ResidualDiagnostics>,
}

fn wald_to_dicts<'py>(
    py: Python<'py>,
    tests: &[WaldTest],
    ddf: &str,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    tests
        .iter()
        .map(|t| {
            let d = PyDict::new(py);
            d.set_item("term", &t.term)?;
            d.set_item("f_statistic", t.f_statistic)?;
            d.set_item("num_df", t.num_df)?;
            d.set_item("den_df", t.den_df)?;
            d.set_item("p_value", t.p_value)?;
            d.set_item("ddf_method", ddf)?;
            Ok(d)
        })
        .collect()
}

#[pymethods]
impl PyFitResult {
    /// Return a formatted summary of the model fit.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Return the variance components as a dict of {name: value}.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Mapping from variance component name (e.g. "genotype", "residual")
    ///     to the estimated sigma^2 value.
    fn variance_components(&self) -> PyResult<HashMap<String, f64>> {
        let mut map = HashMap::new();
        for vc in &self.inner.variance_components {
            if let Some((_, val)) = vc.parameters.first() {
                map.insert(vc.name.clone(), *val);
            }
        }
        Ok(map)
    }

    /// Approximate standard errors of the variance components as a dict
    /// (only available after AI-REML; empty otherwise).
    fn variance_components_se(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        if self.inner.variance_se.iter().any(|se| *se > 0.0) {
            for (vc, se) in self
                .inner
                .variance_components
                .iter()
                .zip(self.inner.variance_se.iter())
            {
                map.insert(vc.name.clone(), *se);
            }
        }
        map
    }

    /// Variance parameters that ended at the boundary of the parameter space
    /// (effectively zero), as a dict of {name: bool} (first parameter of each
    /// component; see variance_parameters() for all of them).
    fn at_boundary(&self) -> HashMap<String, bool> {
        self.inner
            .variance_components
            .iter()
            .map(|vc| {
                (
                    vc.name.clone(),
                    vc.at_boundary.first().copied().unwrap_or(false),
                )
            })
            .collect()
    }

    /// Every variance parameter as a list of dicts with keys component,
    /// structure, name, value, se and at_boundary (structured models have
    /// several parameters per component, e.g. sigma2 and rho for AR1).
    fn variance_parameters<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for vc in &self.inner.variance_components {
            for (i, (name, value)) in vc.parameters.iter().enumerate() {
                let d = PyDict::new(py);
                d.set_item("component", &vc.name)?;
                d.set_item("structure", &vc.structure)?;
                d.set_item("name", name)?;
                d.set_item("value", value)?;
                d.set_item("se", vc.se.get(i).copied().unwrap_or(0.0))?;
                d.set_item(
                    "at_boundary",
                    vc.at_boundary.get(i).copied().unwrap_or(false),
                )?;
                out.push(d);
            }
        }
        Ok(out)
    }

    /// Residual diagnostics as a dict of numpy arrays (fitted, conditional,
    /// marginal, standardized, studentized, leverage, cooks_distance), or
    /// None for structured residuals.
    fn residual_diagnostics<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        let diag = self.diagnostics.as_ref()?;
        let d = PyDict::new(py);
        let arr = |v: &Vec<f64>| PyArray1::from_vec(py, v.clone());
        d.set_item("fitted", arr(&diag.fitted)).ok()?;
        d.set_item("conditional", arr(&diag.conditional)).ok()?;
        d.set_item("marginal", arr(&diag.marginal)).ok()?;
        d.set_item("standardized", arr(&diag.standardized)).ok()?;
        d.set_item("studentized", arr(&diag.studentized)).ok()?;
        d.set_item("leverage", arr(&diag.leverage)).ok()?;
        d.set_item("cooks_distance", arr(&diag.cooks_distance))
            .ok()?;
        Some(d)
    }

    /// Return the fixed effects as a list of (term, level, estimate, se) tuples.
    fn fixed_effects(&self) -> PyResult<Vec<(String, String, f64, f64)>> {
        Ok(self
            .inner
            .fixed_effects
            .iter()
            .map(|e| (e.term.clone(), e.level.clone(), e.estimate, e.se))
            .collect())
    }

    /// Variance-covariance matrix of the fixed effects (p x p numpy array).
    fn fixed_effects_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        PyArray2::from_vec2(py, &self.inner.fixed_cov)
            .map_err(|e| PyValueError::new_err(format!("Failed to build covariance array: {}", e)))
    }

    /// Return the random effects as a dict of {term_name: numpy_array}.
    fn random_effects(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
        let mut map = HashMap::new();
        for block in &self.inner.random_effects {
            let values: Vec<f64> = block.effects.iter().map(|e| e.estimate).collect();
            map.insert(block.term.clone(), PyArray1::from_vec(py, values).unbind());
        }
        Ok(map)
    }

    /// Standard errors (sqrt of prediction error variance) of the random
    /// effects as a dict of {term_name: numpy_array}.
    fn random_effects_se(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
        let mut map = HashMap::new();
        for block in &self.inner.random_effects {
            let values: Vec<f64> = block.effects.iter().map(|e| e.se).collect();
            map.insert(block.term.clone(), PyArray1::from_vec(py, values).unbind());
        }
        Ok(map)
    }

    /// Reliability (1 - PEV/sigma^2) of the random effects as a dict of
    /// {term_name: numpy_array}.
    fn reliabilities(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
        let mut map = HashMap::new();
        for block in &self.inner.random_effects {
            let sigma2 = self.inner.variance_component(&block.term).unwrap_or(0.0);
            let values: Vec<f64> = block
                .effects
                .iter()
                .map(|e| reliability_from_se(e.se, sigma2))
                .collect();
            map.insert(block.term.clone(), PyArray1::from_vec(py, values).unbind());
        }
        Ok(map)
    }

    /// Return the random effect level names for each term.
    fn random_effect_levels(&self) -> PyResult<HashMap<String, Vec<String>>> {
        let mut map = HashMap::new();
        for block in &self.inner.random_effects {
            let levels: Vec<String> = block.effects.iter().map(|e| e.level.clone()).collect();
            map.insert(block.term.clone(), levels);
        }
        Ok(map)
    }

    /// Return the residuals as a numpy array.
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.inner.residuals.clone()))
    }

    /// Wald F-tests for the fixed-effect terms as a list of dicts with keys
    /// term, f_statistic, num_df, den_df, p_value, ddf_method.
    ///
    /// `ddf` is "containment" (n - rank(X), default) or "satterthwaite"
    /// (available for scaled-identity models fitted with AI-REML; falls back
    /// to containment otherwise, see the ddf_method key).
    #[pyo3(signature = (ddf="containment"))]
    fn wald_tests<'py>(&self, py: Python<'py>, ddf: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        match ddf.to_lowercase().as_str() {
            "containment" => wald_to_dicts(py, &wald_tests(&self.inner), "containment"),
            "satterthwaite" => match self.inner.wald_tests_satterthwaite() {
                Some(tests) => wald_to_dicts(py, &tests, "satterthwaite"),
                None => wald_to_dicts(py, &wald_tests(&self.inner), "containment"),
            },
            other => Err(PyValueError::new_err(format!(
                "Unknown ddf method '{}'; use 'containment' or 'satterthwaite'",
                other
            ))),
        }
    }

    /// Return the restricted log-likelihood.
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    /// Return the AIC.
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// Return the BIC.
    fn bic(&self) -> f64 {
        self.inner.bic()
    }

    /// Whether the REML algorithm converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of iterations performed.
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations
    }

    /// Number of observations.
    #[getter]
    fn n_obs(&self) -> usize {
        self.inner.n_obs
    }

    /// Number of fixed effect parameters.
    #[getter]
    fn n_fixed_params(&self) -> usize {
        self.inner.n_fixed_params
    }

    /// Number of variance parameters.
    #[getter]
    fn n_variance_params(&self) -> usize {
        self.inner.n_variance_params
    }

    /// Return the iteration history as a list of dicts with keys
    /// iteration, log_likelihood, change and variance_params.
    fn iteration_history<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .history
            .iter()
            .map(|h| {
                let d = PyDict::new(py);
                d.set_item("iteration", h.iteration)?;
                d.set_item("log_likelihood", h.log_likelihood)?;
                d.set_item("change", h.change)?;
                d.set_item("variance_params", h.variance_params.clone())?;
                Ok(d)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "FitResult(converged={}, n_iter={}, loglik={:.4}, aic={:.4})",
            self.inner.converged,
            self.inner.n_iterations,
            self.inner.log_likelihood,
            self.inner.aic()
        )
    }

    fn __str__(&self) -> String {
        self.inner.summary()
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Compute the A-inverse matrix from a pedigree.
///
/// This is a convenience function that sorts the pedigree (if needed) and
/// computes A-inverse in a single call. The row/column order is
/// `ped.animal_ids()` after sorting.
///
/// Parameters
/// ----------
/// ped : Pedigree
///     The pedigree to compute A-inverse for.
/// inbreeding : bool
///     Account for inbreeding (Meuwissen & Luo 1992). Default True.
///
/// Returns
/// -------
/// tuple
///     (data, indices, indptr, shape) for scipy.sparse.csc_matrix.
#[pyfunction]
#[pyo3(signature = (ped, inbreeding=true))]
fn compute_a_inverse(py: Python<'_>, ped: &mut PyPedigree, inbreeding: bool) -> PyResult<PyObject> {
    if !ped.inner.is_sorted() {
        ped.inner.sort_pedigree().map_err(to_pyerr)?;
    }
    let ainv = if inbreeding {
        core::genetics::compute_a_inverse_with_inbreeding(&ped.inner).map_err(to_pyerr)?
    } else {
        core::genetics::compute_a_inverse(&ped.inner).map_err(to_pyerr)?
    };
    sparse_to_scipy_csc(py, &ainv)
}

/// Compute inbreeding coefficients (Meuwissen & Luo 1992) for all animals of
/// a pedigree, in `ped.animal_ids()` order (the pedigree is sorted if needed).
#[pyfunction]
fn compute_inbreeding<'py>(
    py: Python<'py>,
    ped: &mut PyPedigree,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !ped.inner.is_sorted() {
        ped.inner.sort_pedigree().map_err(to_pyerr)?;
    }
    let f = core::genetics::compute_inbreeding(&ped.inner).map_err(to_pyerr)?;
    Ok(PyArray1::from_vec(py, f))
}

/// Compute the genomic relationship matrix (G) using VanRaden Method 1.
///
/// Parameters
/// ----------
/// markers : numpy.ndarray
///     An (n_individuals x n_markers) matrix with 0/1/2 coding.
/// allele_freqs : numpy.ndarray or None
///     Optional allele frequencies. If None, estimated from marker data.
///
/// Returns
/// -------
/// numpy.ndarray
///     The G-matrix of dimension (n_individuals x n_individuals).
#[pyfunction]
#[pyo3(signature = (markers, allele_freqs=None))]
fn compute_g_matrix<'py>(
    py: Python<'py>,
    markers: PyReadonlyArray2<f64>,
    allele_freqs: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let marker_shape = markers.shape();
    let n = marker_shape[0];
    let m = marker_shape[1];

    // Convert numpy 2D array to nalgebra DMatrix (works for any memory layout).
    let markers_view = markers.as_array();
    let marker_mat = nalgebra::DMatrix::from_fn(n, m, |i, j| markers_view[[i, j]]);

    let freqs_vec: Option<Vec<f64>> = match allele_freqs {
        Some(arr) => Some(
            arr.as_slice()
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to read allele frequency array: {}", e))
                })?
                .to_vec(),
        ),
        None => None,
    };

    let g =
        core::genetics::compute_g_matrix(&marker_mat, freqs_vec.as_deref()).map_err(to_pyerr)?;

    // nalgebra stores column-major, numpy expects row-major, so we build row vectors
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| g[(i, j)]).collect())
        .collect();

    PyArray2::from_vec2(py, &rows)
        .map_err(|e| PyValueError::new_err(format!("Failed to create G-matrix array: {}", e)))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// OpenBLUP native extension: Open-source REML and BLUP for plant and animal breeding.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPedigree>()?;
    m.add_class::<PyMixedModel>()?;
    m.add_class::<PyFitResult>()?;
    m.add_function(wrap_pyfunction!(compute_a_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_inbreeding, m)?)?;
    m.add_function(wrap_pyfunction!(compute_g_matrix, m)?)?;
    Ok(())
}
