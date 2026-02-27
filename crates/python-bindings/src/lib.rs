use std::collections::HashMap;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use plant_breeding_lmm_core as core;
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::genetics::Pedigree;
use plant_breeding_lmm_core::lmm::FitResult;
use plant_breeding_lmm_core::variance::Identity;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a core library error into a Python ValueError.
fn to_pyerr(e: core::LmmError) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

/// Convert a sparse CSC matrix into the (data, indices, indptr, shape) tuple
/// that Python/scipy expects for constructing a `csc_matrix`.
fn sparse_to_scipy_csc<'py>(
    py: Python<'py>,
    mat: &sprs::CsMat<f64>,
) -> PyResult<PyObject> {
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

// ---------------------------------------------------------------------------
// PyPedigree
// ---------------------------------------------------------------------------

/// A pedigree representing parent-offspring relationships.
///
/// Used for computing the additive relationship matrix inverse (A-inverse)
/// needed for pedigree-based BLUP.
#[pyclass(name = "PyPedigree")]
struct PyPedigree {
    inner: Pedigree,
}

#[pymethods]
impl PyPedigree {
    /// Create an empty pedigree.
    #[new]
    fn new() -> Self {
        PyPedigree {
            inner: Pedigree::new(),
        }
    }

    /// Load a pedigree from a CSV file.
    ///
    /// The CSV must have columns: animal, sire, dam.
    /// Unknown parents are coded as "0", "", or "NA".
    #[staticmethod]
    fn from_csv(path: &str) -> PyResult<Self> {
        let ped = Pedigree::from_csv(path).map_err(to_pyerr)?;
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
    fn add_animal(
        &mut self,
        animal: &str,
        sire: Option<&str>,
        dam: Option<&str>,
    ) -> PyResult<()> {
        self.inner.add_animal(animal, sire, dam).map_err(to_pyerr)
    }

    /// Return the number of animals in the pedigree.
    fn n_animals(&self) -> usize {
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

    /// Compute A-inverse (Henderson's rules, no inbreeding).
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

    /// Compute inbreeding coefficients for all animals.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A 1D array of inbreeding coefficients.
    fn compute_inbreeding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = core::genetics::compute_inbreeding(&self.inner).map_err(to_pyerr)?;
        Ok(PyArray1::from_vec(py, f))
    }

    /// Get the list of animal IDs in pedigree order.
    fn animal_ids(&self) -> Vec<String> {
        (0..self.inner.n_animals())
            .map(|i| self.inner.animal_id(i).to_string())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PyMixedModel
// ---------------------------------------------------------------------------

/// A mixed model builder that accumulates data, fixed effects, and random effects,
/// then fits the model via REML.
#[pyclass(name = "PyMixedModel")]
struct PyMixedModel {
    df: Option<DataFrame>,
    response: Option<String>,
    fixed_formula: Option<String>,
    random_terms: Vec<RandomTermPy>,
    max_iter: usize,
    convergence_tol: f64,
}

/// Internal representation of a random term specification for the Python API.
struct RandomTermPy {
    column: String,
    ginv: Option<sprs::CsMat<f64>>,
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
            max_iter: 50,
            convergence_tol: 1e-6,
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
                let data: Vec<f64> = arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("Failed to read array for '{}': {}", name, e))
                })?.to_vec();
                df.add_float_column(name, data).map_err(to_pyerr)?;
                continue;
            }

            // Try list of floats
            if let Ok(vals) = obj.extract::<Vec<f64>>(py) {
                df.add_float_column(name, vals).map_err(to_pyerr)?;
                continue;
            }

            // Try list of ints
            if let Ok(vals) = obj.extract::<Vec<i64>>(py) {
                df.add_integer_column(name, vals).map_err(to_pyerr)?;
                continue;
            }

            // Try numpy i64 array
            if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>(py) {
                let data: Vec<i64> = arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("Failed to read array for '{}': {}", name, e))
                })?.to_vec();
                df.add_integer_column(name, data).map_err(to_pyerr)?;
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
    /// Numeric columns are auto-detected as Float; others become Factor.
    /// Use `as_factor(column)` to convert numeric columns to categorical.
    fn load_csv(&mut self, path: &str) -> PyResult<()> {
        let df = DataFrame::from_csv(path).map_err(to_pyerr)?;
        self.df = Some(df);
        Ok(())
    }

    /// Convert a column to a Factor (categorical) column.
    ///
    /// This is useful when integer-coded columns (like block numbers read from CSV)
    /// should be treated as categorical rather than continuous.
    fn as_factor(&mut self, column: &str) -> PyResult<()> {
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("No data loaded. Call set_data() or load_csv() first."))?;
        df.as_factor(column).map_err(to_pyerr)
    }

    /// Set the response (dependent) variable.
    fn set_response(&mut self, column: &str) -> PyResult<()> {
        self.response = Some(column.to_string());
        Ok(())
    }

    /// Set the fixed effects formula.
    ///
    /// Examples: "mu", "mu + rep", "mu + rep + block"
    /// "mu" or "intercept" or "1" adds an intercept.
    fn add_fixed(&mut self, formula: &str) -> PyResult<()> {
        self.fixed_formula = Some(formula.to_string());
        Ok(())
    }

    /// Add a random effect term.
    ///
    /// Parameters
    /// ----------
    /// column : str
    ///     The factor column to use as the grouping variable.
    /// ginverse : tuple or None
    ///     Optional relationship matrix inverse as (data, indices, indptr, shape)
    ///     from a scipy.sparse.csc_matrix. If None, an identity matrix is used.
    #[pyo3(signature = (column, ginverse=None))]
    fn add_random(
        &mut self,
        py: Python<'_>,
        column: &str,
        ginverse: Option<PyObject>,
    ) -> PyResult<()> {
        let ginv = match ginverse {
            Some(obj) => {
                let tuple = obj.extract::<(
                    PyReadonlyArray1<f64>,
                    PyReadonlyArray1<i64>,
                    PyReadonlyArray1<i64>,
                    (usize, usize),
                )>(py)?;

                let (data_arr, indices_arr, indptr_arr, shape) = tuple;

                let data: Vec<f64> = data_arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("Failed to read ginverse data: {}", e))
                })?.to_vec();
                let indices: Vec<usize> = indices_arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("Failed to read ginverse indices: {}", e))
                })?.iter().map(|&i| i as usize).collect();
                let indptr: Vec<usize> = indptr_arr.as_slice().map_err(|e| {
                    PyValueError::new_err(format!("Failed to read ginverse indptr: {}", e))
                })?.iter().map(|&i| i as usize).collect();

                let mat = sprs::CsMat::new_csc(
                    (shape.0, shape.1),
                    indptr,
                    indices,
                    data,
                );

                Some(mat)
            }
            None => None,
        };

        self.random_terms.push(RandomTermPy {
            column: column.to_string(),
            ginv,
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

    /// Fit the model using REML.
    ///
    /// Returns
    /// -------
    /// PyFitResult
    ///     The fitted model results.
    fn fit(&mut self) -> PyResult<PyFitResult> {
        let df = self
            .df
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No data loaded. Call set_data() or load_csv() first."))?;

        let response = self
            .response
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No response variable set. Call set_response() first."))?;

        let mut builder = core::model::MixedModelBuilder::new()
            .data(df)
            .response(response)
            .max_iterations(self.max_iter)
            .convergence(self.convergence_tol);

        if let Some(ref formula) = self.fixed_formula {
            builder = builder.fixed(formula);
        }

        for rt in &self.random_terms {
            builder = builder.random(&rt.column, Identity::new(1.0), rt.ginv.clone());
        }

        let mut model = builder.build().map_err(to_pyerr)?;
        let result = model.fit_reml().map_err(to_pyerr)?;

        Ok(PyFitResult { inner: result })
    }
}

// ---------------------------------------------------------------------------
// PyFitResult
// ---------------------------------------------------------------------------

/// The result of fitting a mixed model via REML.
#[pyclass(name = "PyFitResult")]
struct PyFitResult {
    inner: FitResult,
}

#[pymethods]
impl PyFitResult {
    /// Print a formatted summary of the model fit.
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
            // The first parameter is typically sigma^2.
            if let Some((_, val)) = vc.parameters.first() {
                map.insert(vc.name.clone(), *val);
            }
        }
        Ok(map)
    }

    /// Return the fixed effects as a list of (term, level, estimate, se) tuples.
    fn fixed_effects(&self) -> PyResult<Vec<(String, String, f64, f64)>> {
        let effects: Vec<(String, String, f64, f64)> = self
            .inner
            .fixed_effects
            .iter()
            .map(|e| (e.term.clone(), e.level.clone(), e.estimate, e.se))
            .collect();
        Ok(effects)
    }

    /// Return the random effects as a dict of {term_name: numpy_array}.
    fn random_effects<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<HashMap<String, Py<PyArray1<f64>>>> {
        let mut map = HashMap::new();
        for block in &self.inner.random_effects {
            let values: Vec<f64> = block.effects.iter().map(|e| e.estimate).collect();
            let arr = PyArray1::from_vec(py, values).unbind();
            map.insert(block.term.clone(), arr);
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

    /// Return the iteration history as a list of dicts.
    fn iteration_history(&self) -> Vec<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.inner
                .history
                .iter()
                .map(|h| {
                    let mut m = HashMap::new();
                    m.insert("iteration".to_string(), h.iteration.into_pyobject(py).unwrap().into_any().unbind());
                    m.insert("log_likelihood".to_string(), h.log_likelihood.into_pyobject(py).unwrap().into_any().unbind());
                    m.insert("change".to_string(), h.change.into_pyobject(py).unwrap().into_any().unbind());
                    m
                })
                .collect()
        })
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
/// This is a convenience function that sorts the pedigree and computes A-inverse
/// in a single call.
///
/// Parameters
/// ----------
/// ped : PyPedigree
///     The pedigree to compute A-inverse for.
///
/// Returns
/// -------
/// tuple
///     (data, indices, indptr, shape) for scipy.sparse.csc_matrix.
#[pyfunction]
fn compute_a_inverse(py: Python<'_>, ped: &mut PyPedigree) -> PyResult<PyObject> {
    if !ped.inner.is_sorted() {
        ped.inner.sort_pedigree().map_err(to_pyerr)?;
    }
    let ainv = core::genetics::compute_a_inverse(&ped.inner).map_err(to_pyerr)?;
    sparse_to_scipy_csc(py, &ainv)
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

    // Convert numpy 2D array to nalgebra DMatrix
    let marker_slice = markers.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Failed to read marker array: {}", e))
    })?;
    let marker_mat = nalgebra::DMatrix::from_row_slice(n, m, marker_slice);

    let freqs_vec: Option<Vec<f64>> = allele_freqs.map(|arr| {
        arr.as_slice()
            .expect("Failed to read allele frequency array")
            .to_vec()
    });

    let g = core::genetics::compute_g_matrix(
        &marker_mat,
        freqs_vec.as_deref(),
    )
    .map_err(to_pyerr)?;

    // Convert nalgebra DMatrix to numpy 2D array
    // nalgebra stores column-major, numpy expects row-major, so we build row vectors
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| g[(i, j)]).collect())
        .collect();

    let arr = PyArray2::from_vec2(py, &rows)
        .map_err(|e| PyValueError::new_err(format!("Failed to create G-matrix array: {}", e)))?;

    Ok(arr)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// OpenBLUP: Open-source REML and BLUP for plant and animal breeding.
#[pymodule]
fn openblup(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPedigree>()?;
    m.add_class::<PyMixedModel>()?;
    m.add_class::<PyFitResult>()?;
    m.add_function(wrap_pyfunction!(compute_a_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_g_matrix, m)?)?;
    Ok(())
}
