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

    /// The principal square root of a symmetric positive semidefinite matrix,
    /// with the negative eigenvalue it set to zero past the floor of `tolerance`.
    ///
    /// An eigenvalue below zero by no more than `tolerance`, relative to the
    /// largest entry and per row, is rounding noise of a singular matrix and
    /// counts as zero. Each entry is known to `tolerance` of the largest at best,
    /// and by Weyl's inequality errors of that size move an eigenvalue by at most
    /// the size of the matrix times that, so a fixed floor would refuse large
    /// singular matrices for their rounding alone.
    ///
    /// `band` is for a matrix that is positive semidefinite in exact arithmetic
    /// and whose entries the caller knows to within a wider bound: that bound,
    /// relative to the largest entry, so that `band` per row is the most rounding
    /// can have moved an eigenvalue. An eigenvalue below the floor of `tolerance`
    /// and above that of `band` counts as zero too, and the most negative such is
    /// returned. With `band` equal to `tolerance` nothing is returned.
    ///
    /// # Errors
    ///
    /// Returns [`MMError::InvalidConfiguration`] if an eigenvalue is below the
    /// floor of `band`, or of `tolerance` when that is wider, and whatever
    /// [`Matrix::symmetric_eigen`] returns.
    pub(super) fn principal_sqrt(
        &self,
        tolerance: f64,
        band: f64,
    ) -> MMResult<(Self, Option<Clipped>)> {
        let (values, vectors) = self.symmetric_eigen()?;
        let rows = self.size.max(1) as f64;
        let largest = self.max_abs();
        let floor = -tolerance * rows;
        let band_floor = -band.max(tolerance) * rows;
        let lowest = values.iter().copied().fold(f64::INFINITY, f64::min);
        if lowest < band_floor * largest {
            return Err(MMError::InvalidConfiguration(format!(
                "matrix is not positive semidefinite: eigenvalue {lowest} is below the floor {}",
                band_floor * largest
            )));
        }
        let clipped = (lowest < floor * largest).then(|| Clipped {
            eigenvalue: lowest / largest,
            floor,
            band: band_floor,
        });
        let mut root = Self::zeros(self.size);
        for (index, value) in values.iter().enumerate() {
            let scale = value.max(0.0).sqrt();
            for row in 0..self.size {
                for column in 0..self.size {
                    let added = scale * vectors.at(row, index) * vectors.at(column, index);
                    root.set(row, column, root.at(row, column) + added);
                }
            }
        }
        Ok((root, clipped))
    }
}

/// A negative eigenvalue that [`Matrix::principal_sqrt`] set to zero below the
/// floor of its `tolerance`. Each value is relative to the largest entry of the
/// matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    not(all(feature = "options", feature = "margin")),
    allow(dead_code, reason = "only the option quoter reports a clip")
)]
pub(super) struct Clipped {
    /// The most negative eigenvalue.
    pub(super) eigenvalue: f64,
    /// The floor of the tolerance, which it is below.
    pub(super) floor: f64,
    /// The floor of the band, which it is above.
    pub(super) band: f64,
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
        assert!(indefinite.principal_sqrt(1e-12, 1e-12).is_err());
    }

    #[test]
    fn test_a_negative_eigenvalue_within_the_floor_counts_as_zero_and_past_it_does_not() {
        let at = |eigenvalue: f64| {
            let mut diagonal = Matrix::zeros(4);
            diagonal.set(0, 0, 1.0);
            diagonal.set(1, 1, 1.0);
            diagonal.set(2, 2, 1.0);
            diagonal.set(3, 3, eigenvalue);
            diagonal.principal_sqrt(1e-12, 1e-12)
        };
        // Four rows, so a floor of 4e-12 of the largest entry.
        let (within, clipped) = at(-3e-12).unwrap();
        assert_eq!(clipped, None);
        assert_eq!(within.at(3, 3), 0.0);
        assert!(at(-5e-12).is_err());
    }

    /// Three eigenvalues of 2 and one given, so the largest entry is 2.
    fn with_eigenvalue(eigenvalue: f64) -> Matrix {
        let mut diagonal = Matrix::zeros(4);
        for index in 0..3 {
            diagonal.set(index, index, 2.0);
        }
        diagonal.set(3, 3, eigenvalue);
        diagonal
    }

    #[test]
    fn test_a_negative_eigenvalue_past_the_floor_and_within_the_band_is_clipped_and_returned() {
        // Floors of 4e-12 and 4e-11 per row of four, of the largest entry of 2.
        let (root, clipped) = with_eigenvalue(-2e-11)
            .principal_sqrt(1e-12, 1e-11)
            .unwrap();
        assert_eq!(root.at(3, 3), 0.0);
        assert_eq!(root.at(0, 0), 2.0_f64.sqrt());
        assert_eq!(
            clipped,
            Some(Clipped {
                eigenvalue: -1e-11,
                floor: -4e-12,
                band: -4e-11,
            })
        );
        // The same matrix without the band is refused.
        assert!(
            with_eigenvalue(-2e-11)
                .principal_sqrt(1e-12, 1e-12)
                .is_err()
        );
    }

    #[test]
    fn test_a_negative_eigenvalue_past_the_band_is_refused_and_named() {
        let error = with_eigenvalue(-1e-10)
            .principal_sqrt(1e-12, 1e-11)
            .unwrap_err();
        let MMError::InvalidConfiguration(message) = error else {
            panic!("{error:?}");
        };
        assert!(
            message.contains(&format!("eigenvalue {}", -1e-10_f64)),
            "{message}"
        );
        assert!(
            message.contains(&format!("floor {}", -8e-11_f64)),
            "{message}"
        );
    }

    #[test]
    fn test_no_eigenvalue_past_the_floor_returns_no_clip() {
        let band = 1e-6;
        let (_, definite) = matrix(&[&[4.0, 1.0], &[1.0, 3.0]])
            .principal_sqrt(1e-12, band)
            .unwrap();
        assert_eq!(definite, None);
        let (_, singular) = matrix(&[&[1.0, 1.0], &[1.0, 1.0]])
            .principal_sqrt(1e-12, band)
            .unwrap();
        assert_eq!(singular, None);
        // Negative, but within the floor of the tolerance: rounding as before.
        let (_, within) = with_eigenvalue(-7e-12).principal_sqrt(1e-12, band).unwrap();
        assert_eq!(within, None);
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
        let (root, _) = low_rank.principal_sqrt(1e-12, 1e-12).unwrap();
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
        let (root, _) = matrix(&[&[4.0, 1.0], &[1.0, 3.0]])
            .principal_sqrt(1e-12, 1e-12)
            .unwrap();
        let det_root = 11.0_f64.sqrt();
        let norm = (7.0 + 2.0 * det_root).sqrt();
        assert!((root.at(0, 0) - (4.0 + det_root) / norm).abs() < 1e-14);
        assert!((root.at(0, 1) - 1.0 / norm).abs() < 1e-14);
        assert!((root.at(1, 1) - (3.0 + det_root) / norm).abs() < 1e-14);
    }

    #[test]
    fn test_matrix_sqrt_of_singular_matrix_keeps_null_direction() {
        let (root, _) = matrix(&[&[1.0, 1.0], &[1.0, 1.0]])
            .principal_sqrt(1e-12, 1e-12)
            .unwrap();
        let half_root = 1.0 / 2.0_f64.sqrt();
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!((root.at(i, j) - half_root).abs() < 1e-12);
        }
    }

    #[test]
    fn test_matrix_sqrt_of_indefinite_matrix_is_refused() {
        let error = matrix(&[&[1.0, 2.0], &[2.0, 1.0]])
            .principal_sqrt(1e-12, 1e-12)
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
