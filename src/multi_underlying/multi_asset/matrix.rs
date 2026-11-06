//! Dense symmetric matrices in `f64`, for the construction of the risk matrix.
//!
//! The principal square root of a symmetric matrix has no closed form beyond
//! two dimensions and needs an iterative eigendecomposition, which is why this
//! is `f64`: it runs once, at construction, and its result reaches the quotes
//! only through a checked conversion to `Decimal`.

use crate::types::error::{MMError, MMResult};

/// Cyclic Jacobi sweeps allowed. Convergence is quadratic: a matrix of the
/// supported size needs well under ten.
const MAX_SWEEPS: usize = 64;

/// Off diagonal mass, relative to the whole matrix and per row, at which a sweep stops. Rounding
/// leaves mass of the order of the machine's epsilon on each row, so a tolerance that does not
/// grow with the size cannot be met by a large matrix, however well it has converged.
const OFF_DIAGONAL_TOLERANCE: f64 = 1e-15;

/// A square matrix; indices out of range read as `NaN` and ignore writes, and
/// the `NaN` reaches the finite checks of the caller.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Matrix {
    size: usize,
    cells: Vec<f64>,
}

impl Matrix {
    /// A zero matrix. `size` is bounded by the caller, so its square fits.
    pub(super) fn zeros(size: usize) -> Self {
        Self {
            size,
            cells: vec![0.0; size.saturating_mul(size)],
        }
    }

    fn cell(&self, row: usize, column: usize) -> usize {
        row.saturating_mul(self.size).saturating_add(column)
    }

    #[must_use]
    pub(super) fn at(&self, row: usize, column: usize) -> f64 {
        self.cells
            .get(self.cell(row, column))
            .copied()
            .unwrap_or(f64::NAN)
    }

    pub(super) fn set(&mut self, row: usize, column: usize, value: f64) {
        let cell = self.cell(row, column);
        if let Some(slot) = self.cells.get_mut(cell) {
            *slot = value;
        }
    }

    /// The largest absolute entry.
    #[must_use]
    pub(super) fn max_abs(&self) -> f64 {
        self.cells.iter().fold(0.0, |max, cell| cell.abs().max(max))
    }

    /// Squared Frobenius norm of the entries off the diagonal.
    fn off_diagonal(&self) -> f64 {
        let mut sum = 0.0;
        for row in 0..self.size {
            for column in 0..self.size {
                if row != column {
                    sum += self.at(row, column).powi(2);
                }
            }
        }
        sum
    }

    /// Eigenvalues and eigenvectors (by column) of a symmetric matrix, by
    /// cyclic Jacobi rotations.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::NumericalError`] if an entry is not finite or the
    /// rotations do not converge.
    pub(super) fn symmetric_eigen(&self) -> MMResult<(Vec<f64>, Matrix)> {
        if self.cells.iter().any(|cell| !cell.is_finite()) {
            return Err(MMError::NumericalError(
                "risk matrix has a non finite entry".to_string(),
            ));
        }
        let mut work = self.clone();
        let mut vectors = Self::zeros(self.size);
        for index in 0..self.size {
            vectors.set(index, index, 1.0);
        }
        let threshold = (self.max_abs() * OFF_DIAGONAL_TOLERANCE * self.size.max(1) as f64).powi(2);
        for _ in 0..MAX_SWEEPS {
            if work.off_diagonal() <= threshold {
                break;
            }
            for p in 0..self.size {
                for q in p.saturating_add(1)..self.size {
                    work.rotate(&mut vectors, p, q);
                }
            }
        }
        if work.off_diagonal() > threshold {
            return Err(MMError::NumericalError(format!(
                "risk matrix eigendecomposition did not converge in {MAX_SWEEPS} sweeps"
            )));
        }
        let values = (0..self.size).map(|index| work.at(index, index)).collect();
        Ok((values, vectors))
    }

    /// One Jacobi rotation annihilating `(p, q)`.
    fn rotate(&mut self, vectors: &mut Self, p: usize, q: usize) {
        let off = self.at(p, q);
        if off == 0.0 {
            return;
        }
        let theta = (self.at(q, q) - self.at(p, p)) / (2.0 * off);
        let tangent = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        // `signum` of zero is one in `f64`, the wanted rotation of 45 degrees.
        let cosine = 1.0 / (tangent * tangent + 1.0).sqrt();
        let sine = tangent * cosine;
        for index in 0..self.size {
            let (ip, iq) = (self.at(index, p), self.at(index, q));
            self.set(index, p, cosine * ip - sine * iq);
            self.set(index, q, sine * ip + cosine * iq);
        }
        for index in 0..self.size {
            let (pi, qi) = (self.at(p, index), self.at(q, index));
            self.set(p, index, cosine * pi - sine * qi);
            self.set(q, index, sine * pi + cosine * qi);
        }
        for index in 0..self.size {
            let (ip, iq) = (vectors.at(index, p), vectors.at(index, q));
            vectors.set(index, p, cosine * ip - sine * iq);
            vectors.set(index, q, sine * ip + cosine * iq);
        }
    }

    /// The principal square root of a symmetric positive semidefinite matrix.
    ///
    /// An eigenvalue below zero by no more than `tolerance`, relative to the
    /// largest entry and per row, is rounding noise of a singular matrix and
    /// counts as zero. Each entry is known to `tolerance` of the largest at best,
    /// and by Weyl's inequality errors of that size move an eigenvalue by at most
    /// the size of the matrix times that, so a fixed floor would refuse large
    /// singular matrices for their rounding alone.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if an eigenvalue is more
    /// negative than that, and whatever [`Matrix::symmetric_eigen`] returns.
    pub(super) fn principal_sqrt(&self, tolerance: f64) -> MMResult<Self> {
        let (values, vectors) = self.symmetric_eigen()?;
        let floor = -tolerance * self.size.max(1) as f64 * self.max_abs();
        let mut root = Self::zeros(self.size);
        for (index, value) in values.iter().enumerate() {
            if *value < floor {
                return Err(MMError::InvalidConfiguration(format!(
                    "matrix is not positive semidefinite: eigenvalue {value}"
                )));
            }
            let scale = value.max(0.0).sqrt();
            for row in 0..self.size {
                for column in 0..self.size {
                    let added = scale * vectors.at(row, index) * vectors.at(column, index);
                    root.set(row, column, root.at(row, column) + added);
                }
            }
        }
        Ok(root)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_a_large_indefinite_matrix_is_still_refused() {
        // Rank one, with a negative eigenvalue well past any rounding, on the same scale.
        let size = 64;
        let mut indefinite = Matrix::zeros(size);
        for i in 0..size {
            for j in 0..size {
                indefinite.set(i, j, 1.0);
            }
        }
        indefinite.set(0, 0, 1.0 - 1e-6);
        indefinite.set(1, 1, 1.0 - 1e-6);
        indefinite.set(0, 1, 1.0 + 1e-6);
        indefinite.set(1, 0, 1.0 + 1e-6);
        assert!(indefinite.principal_sqrt(1e-12).is_err());
    }

    #[test]
    fn test_a_negative_eigenvalue_within_the_floor_counts_as_zero_and_past_it_does_not() {
        let at = |eigenvalue: f64| {
            let mut diagonal = Matrix::zeros(4);
            diagonal.set(0, 0, 1.0);
            diagonal.set(1, 1, 1.0);
            diagonal.set(2, 2, 1.0);
            diagonal.set(3, 3, eigenvalue);
            diagonal.principal_sqrt(1e-12)
        };
        // Four rows, so a floor of 4e-12 of the largest entry.
        let within = at(-3e-12).unwrap();
        assert_eq!(within.at(3, 3), 0.0);
        assert!(at(-5e-12).is_err());
    }

    #[test]
    fn test_a_large_singular_matrix_converges_to_its_square_root() {
        // Two factors and the largest size a model takes.
        let size = 64;
        let load = |i: usize| (1.0 + i as f64 * 0.37, (i as f64 * 0.91).sin());
        let mut low_rank = Matrix::zeros(size);
        for i in 0..size {
            for j in 0..size {
                let (a, b) = (load(i), load(j));
                low_rank.set(i, j, a.0 * b.0 + a.1 * b.1);
            }
        }
        let root = low_rank.principal_sqrt(1e-12).unwrap();
        let scale = low_rank.max_abs();
        for i in 0..size {
            for j in 0..size {
                let square: f64 = (0..size).map(|k| root.at(i, k) * root.at(k, j)).sum();
                assert!(
                    (square - low_rank.at(i, j)).abs() < 1e-11 * scale,
                    "{i} {j}: {square} against {}",
                    low_rank.at(i, j)
                );
            }
        }
    }

    fn matrix(rows: &[&[f64]]) -> Matrix {
        let mut out = Matrix::zeros(rows.len());
        for (i, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                out.set(i, j, *value);
            }
        }
        out
    }

    #[test]
    fn test_matrix_sqrt_of_two_by_two_matches_closed_form() {
        // sqrt(M) = (M + sqrt(det) I) / sqrt(trace + 2 sqrt(det)).
        let root = matrix(&[&[4.0, 1.0], &[1.0, 3.0]])
            .principal_sqrt(1e-12)
            .unwrap();
        let det_root = 11.0_f64.sqrt();
        let norm = (7.0 + 2.0 * det_root).sqrt();
        assert!((root.at(0, 0) - (4.0 + det_root) / norm).abs() < 1e-14);
        assert!((root.at(0, 1) - 1.0 / norm).abs() < 1e-14);
        assert!((root.at(1, 1) - (3.0 + det_root) / norm).abs() < 1e-14);
    }

    #[test]
    fn test_matrix_sqrt_of_singular_matrix_keeps_null_direction() {
        let root = matrix(&[&[1.0, 1.0], &[1.0, 1.0]])
            .principal_sqrt(1e-12)
            .unwrap();
        let half_root = 1.0 / 2.0_f64.sqrt();
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!((root.at(i, j) - half_root).abs() < 1e-12);
        }
    }

    #[test]
    fn test_matrix_sqrt_of_indefinite_matrix_is_refused() {
        let error = matrix(&[&[1.0, 2.0], &[2.0, 1.0]])
            .principal_sqrt(1e-12)
            .unwrap_err();
        assert!(matches!(error, MMError::InvalidConfiguration(_)));
    }

    #[test]
    fn test_matrix_eigen_with_non_finite_entry_is_refused() {
        let error = matrix(&[&[1.0, f64::NAN], &[f64::NAN, 1.0]])
            .symmetric_eigen()
            .unwrap_err();
        assert!(matches!(error, MMError::NumericalError(_)));
    }

    #[test]
    fn test_matrix_eigen_reconstructs_the_input() {
        let input = matrix(&[&[2.0, -1.0, 0.5], &[-1.0, 3.0, 0.25], &[0.5, 0.25, 1.0]]);
        let (values, vectors) = input.symmetric_eigen().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let rebuilt: f64 = (0..3)
                    .map(|k| values[k] * vectors.at(i, k) * vectors.at(j, k))
                    .sum();
                assert!((rebuilt - input.at(i, j)).abs() < 1e-13);
            }
        }
    }
}
