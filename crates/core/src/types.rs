/// The scalar type used throughout the library.
pub type Scalar = f64;

/// Dense matrix type (column-major).
pub type DenseMatrix = nalgebra::DMatrix<Scalar>;

/// Dense vector type.
pub type DenseVector = nalgebra::DVector<Scalar>;

/// Sparse matrix type (CSC format).
pub type SparseMat = sprs::CsMat<Scalar>;

/// Sparse vector type.
pub type SparseVec = sprs::CsVec<Scalar>;
