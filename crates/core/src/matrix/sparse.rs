use sprs::{CsMat, TriMat};

/// Incremental sparse matrix builder using triplet (COO) format.
///
/// Duplicate entries at the same (row, col) are summed when converting to CSC,
/// which is the natural behavior for MME assembly where contributions are
/// accumulated.
#[derive(Debug)]
pub struct TripletBuilder {
    triplet: TriMat<f64>,
}

impl TripletBuilder {
    /// Create a new builder for a matrix of the given dimensions.
    pub fn new(nrow: usize, ncol: usize) -> Self {
        Self {
            triplet: TriMat::new((nrow, ncol)),
        }
    }

    /// Add a value at (row, col). Duplicate entries will be summed.
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        self.triplet.add_triplet(row, col, val);
    }

    /// Add a symmetric entry: inserts at both (row, col) and (col, row).
    /// If row == col, only one entry is added.
    pub fn add_symmetric(&mut self, row: usize, col: usize, val: f64) {
        self.triplet.add_triplet(row, col, val);
        if row != col {
            self.triplet.add_triplet(col, row, val);
        }
    }

    /// Convert to a CSC (Compressed Sparse Column) matrix.
    /// Duplicate entries are summed.
    pub fn to_csc(&self) -> CsMat<f64> {
        self.triplet.to_csc()
    }

    /// Convert to a CSR (Compressed Sparse Row) matrix.
    pub fn to_csr(&self) -> CsMat<f64> {
        self.triplet.to_csr()
    }

    /// Number of rows.
    pub fn nrow(&self) -> usize {
        self.triplet.rows()
    }

    /// Number of columns.
    pub fn ncol(&self) -> usize {
        self.triplet.cols()
    }
}

/// Create a sparse identity matrix of dimension n in CSC format.
pub fn sparse_identity(n: usize) -> CsMat<f64> {
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csc()
}

/// Create a sparse diagonal matrix from a vector of diagonal values.
pub fn sparse_diagonal(diag: &[f64]) -> CsMat<f64> {
    let n = diag.len();
    let mut tri = TriMat::new((n, n));
    for (i, &val) in diag.iter().enumerate() {
        tri.add_triplet(i, i, val);
    }
    tri.to_csc()
}

/// Multiply a sparse matrix by a dense vector: result = A * x.
pub fn spmv(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrow = a.rows();
    let mut result = vec![0.0; nrow];
    spmv_into(a, x, &mut result);
    result
}

/// Multiply a sparse matrix by a dense vector into an existing buffer.
/// result = A * x (buffer must have length == a.rows()).
pub fn spmv_into(a: &CsMat<f64>, x: &[f64], result: &mut [f64]) {
    assert_eq!(a.cols(), x.len());
    assert_eq!(a.rows(), result.len());

    result.fill(0.0);

    // Iterate over non-zero entries. CsMat is CSC by default.
    for (val, (row, col)) in a.iter() {
        result[row] += val * x[col];
    }
}

/// Compute X' * X for a sparse matrix X, returning a dense matrix.
/// This is used for small fixed-effects blocks.
pub fn xtx_dense(x: &CsMat<f64>) -> nalgebra::DMatrix<f64> {
    let ncol = x.cols();
    let mut result = nalgebra::DMatrix::zeros(ncol, ncol);

    // For each pair of columns, compute their dot product
    let x_csc = if x.is_csc() {
        x.clone()
    } else {
        x.to_csc()
    };

    for j in 0..ncol {
        let col_j = x_csc.outer_view(j);
        if let Some(col_j) = col_j {
            for i in j..ncol {
                let col_i = x_csc.outer_view(i);
                if let Some(col_i) = col_i {
                    let dot = col_j.dot(&col_i);
                    result[(i, j)] = dot;
                    if i != j {
                        result[(j, i)] = dot;
                    }
                }
            }
        }
    }

    result
}

/// Compute X' * y for a sparse matrix X and dense vector y.
pub fn xt_y(x: &CsMat<f64>, y: &[f64]) -> Vec<f64> {
    let ncol = x.cols();
    let mut result = vec![0.0; ncol];

    for (val, (row, col)) in x.iter() {
        result[col] += val * y[row];
    }

    result
}

/// Compute y'Py = y'R_inv*y - sol'*rhs (from MME solution).
/// This is more numerically stable than computing Py first.
pub fn quadratic_form(y_r_inv_y: f64, sol: &[f64], rhs: &[f64]) -> f64 {
    let sol_rhs: f64 = sol.iter().zip(rhs.iter()).map(|(s, r)| s * r).sum();
    y_r_inv_y - sol_rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_builder_basic() {
        let mut builder = TripletBuilder::new(3, 3);
        builder.add(0, 0, 1.0);
        builder.add(1, 1, 2.0);
        builder.add(2, 2, 3.0);
        let mat = builder.to_csc();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.nnz(), 3);
    }

    #[test]
    fn test_triplet_duplicate_summing() {
        let mut builder = TripletBuilder::new(2, 2);
        builder.add(0, 0, 1.5);
        builder.add(0, 0, 2.5); // should sum to 4.0
        let mat = builder.to_csc();
        // The value at (0,0) should be 4.0
        let dense: Vec<f64> = spmv(&mat, &[1.0, 0.0]);
        assert!((dense[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_symmetric() {
        let mut builder = TripletBuilder::new(3, 3);
        builder.add_symmetric(0, 1, 5.0);
        builder.add_symmetric(2, 2, 3.0); // diagonal: should not double
        let mat = builder.to_csc();

        let x = vec![1.0, 1.0, 1.0];
        let result = spmv(&mat, &x);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_identity() {
        let eye = sparse_identity(4);
        assert_eq!(eye.rows(), 4);
        assert_eq!(eye.cols(), 4);
        assert_eq!(eye.nnz(), 4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = spmv(&eye, &x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_sparse_diagonal() {
        let d = sparse_diagonal(&[2.0, 3.0, 5.0]);
        let x = vec![1.0, 1.0, 1.0];
        let result = spmv(&d, &x);
        assert_eq!(result, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_spmv() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        let mut builder = TripletBuilder::new(2, 3);
        builder.add(0, 0, 1.0);
        builder.add(0, 2, 2.0);
        builder.add(1, 1, 3.0);
        let mat = builder.to_csc();

        let x = vec![1.0, 2.0, 3.0];
        let result = spmv(&mat, &x);
        assert!((result[0] - 7.0).abs() < 1e-10); // 1*1 + 0*2 + 2*3 = 7
        assert!((result[1] - 6.0).abs() < 1e-10); // 0*1 + 3*2 + 0*3 = 6
    }

    #[test]
    fn test_xt_y() {
        // X is 3x2: [[1, 0], [1, 0], [0, 1]]
        let mut builder = TripletBuilder::new(3, 2);
        builder.add(0, 0, 1.0);
        builder.add(1, 0, 1.0);
        builder.add(2, 1, 1.0);
        let x = builder.to_csc();

        let y = vec![5.0, 3.0, 7.0];
        let result = xt_y(&x, &y);
        assert!((result[0] - 8.0).abs() < 1e-10); // 1*5 + 1*3 = 8
        assert!((result[1] - 7.0).abs() < 1e-10); // 1*7 = 7
    }

    #[test]
    fn test_quadratic_form() {
        let y_r_inv_y = 100.0;
        let sol = vec![2.0, 3.0, 4.0];
        let rhs = vec![5.0, 6.0, 7.0];
        // sol'*rhs = 10 + 18 + 28 = 56
        let qf = quadratic_form(y_r_inv_y, &sol, &rhs);
        assert!((qf - 44.0).abs() < 1e-10);
    }
}
