use std::num::NonZeroUsize;

use crate::float::FloatExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Index {
    pub row: usize,
    pub col: usize,
}
impl Index {
    pub(crate) fn to_1(self, cols: NonZeroUsize) -> usize {
        self.row * cols.get() + self.col
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: NonZeroUsize,
    data: Vec<f64>,
}
impl Matrix {
    pub fn new(rows: NonZeroUsize, data: Vec<f64>) -> Self {
        if data.len() % rows.get() != 0 {
            panic!("not even");
        }
        if data.is_empty() {
            panic!("no data");
        }
        Self { rows, data }
    }

    pub fn zero(rows: NonZeroUsize, cols: NonZeroUsize) -> Self {
        let data = vec![0.; rows.get() * cols.get()];
        Self::new(rows, data)
    }

    pub fn identity(rows: NonZeroUsize) -> Self {
        let mut matrix = Self::zero(rows, rows);
        for row in 0..rows.get() {
            let index = Index { row, col: row };
            matrix.set_cell(index, 1.);
        }
        matrix
    }

    pub fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    pub fn cols(&self) -> NonZeroUsize {
        let cols = self.data.len() / self.rows.get();
        NonZeroUsize::new(cols).unwrap()
    }

    pub fn is_square(&self) -> bool {
        self.full_partial().is_square()
    }

    pub fn cell(&self, index: Index) -> f64 {
        let index = self.index_2_to_1(index);
        self.data[index]
    }

    pub fn set_cell(&mut self, index: Index, value: f64) {
        let index = self.index_2_to_1(index);
        self.data[index] = value;
    }

    fn index_2_to_1(&self, index: Index) -> usize {
        if self.cols().get() <= index.col {
            panic!("col out of range");
        }
        if self.rows().get() <= index.row {
            panic!("row out of range");
        }
        index.to_1(self.cols())
    }

    pub fn add_scalar(&mut self, value: f64) {
        for cell in &mut self.data {
            *cell += value;
        }
    }

    pub fn mul_scalar(&mut self, value: f64) {
        for cell in &mut self.data {
            *cell *= value;
        }
    }

    pub fn add_matrix(&mut self, other: &Matrix) {
        self.assert_same_shape(other);
        self.data
            .iter_mut()
            .zip(other.data.iter().copied())
            .for_each(|(this, other)| *this += other);
    }

    pub fn assert_same_shape(&self, other: &Matrix) {
        if self.rows != other.rows {
            panic!("unmatched rows");
        }
        if self.data.len() != other.data.len() {
            panic!("unmatched cols");
        }
    }

    pub fn transpose(&self) -> Self {
        self.full_partial().transpose()
    }

    pub fn determinant(&self) -> f64 {
        self.full_partial().determinant()
    }

    pub fn inverse(&self) -> Self {
        self.full_partial().inverse()
    }

    fn full_partial(&self) -> PartialMatrix<'_> {
        let start = Index { row: 0, col: 0 };
        let end = Index {
            row: self.rows().get(),
            col: self.cols().get(),
        };
        PartialMatrix::new(self, start, end)
    }

    pub fn closes_to(&self, other: &Self) -> bool {
        self.assert_same_shape(other);
        for (a, b) in self.data.iter().copied().zip(other.data.iter().copied()) {
            if !a.closes_to(b) {
                return false;
            }
        }
        true
    }

    pub fn mul_matrix(&self, other: &Self) -> Matrix {
        self.full_partial().mul_matrix(other.full_partial())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PartialMatrix<'orig> {
    orig_matrix: &'orig Matrix,
    start: Index,
    /// exclusive
    end: Index,
}
impl<'orig> PartialMatrix<'orig> {
    pub fn new(matrix: &'orig Matrix, start: Index, end: Index) -> Self {
        let start_in_bound = start.row < matrix.rows().get() && start.col < matrix.cols().get();
        let end_in_bound = end.row <= matrix.rows().get() && end.col <= matrix.cols().get();
        let start_end_in_order = start.row < end.row && start.col < end.col;
        let valid = start_in_bound && end_in_bound && start_end_in_order;
        if !valid {
            panic!("partial matrix invalid");
        }
        Self {
            orig_matrix: matrix,
            start,
            end,
        }
    }
}
impl PartialMatrix<'_> {
    pub fn rows(&self) -> NonZeroUsize {
        let rows = self.end.row - self.start.row;
        NonZeroUsize::new(rows).unwrap()
    }

    pub fn cols(&self) -> NonZeroUsize {
        let cols = self.end.col - self.start.col;
        NonZeroUsize::new(cols).unwrap()
    }

    pub fn cell(&self, index: Index) -> f64 {
        let row = index.row + self.start.row;
        let col = index.col + self.start.col;
        let index = Index { row, col };
        self.orig_matrix.cell(index)
    }

    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    pub fn transpose(&self) -> Matrix {
        let mut matrix = Matrix::zero(self.cols(), self.rows());
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let value = self.cell(Index { row, col });
                let index = Index { row: col, col: row };
                matrix.set_cell(index, value);
            }
        }
        matrix
    }

    pub fn determinant(&self) -> f64 {
        if !self.is_square() {
            panic!("not a square matrix");
        }
        if self.rows().get() == 1 {
            return self.cell(Index { row: 0, col: 0 });
        }
        if self.rows().get() == 2 {
            return self.cell(Index { row: 0, col: 0 }) * self.cell(Index { row: 1, col: 1 })
                - self.cell(Index { row: 0, col: 1 }) * self.cell(Index { row: 1, col: 0 });
        }

        let mut sum = 0.;
        let mut alt_sign = 1.;
        for col in 0..self.cols().get() {
            let index = Index { row: 0, col };
            let value = self.cell(index);
            let matrix = self.exclude_cross(index);
            let det = matrix.determinant();
            sum += value * det * alt_sign;
            alt_sign *= -1.;
        }

        sum
    }

    pub fn exclude_cross(&self, index: Index) -> Matrix {
        let _valid = self.cell(index);
        let rows = self.rows().get() - 1;
        let cols = self.cols().get() - 1;
        let Some(rows) = NonZeroUsize::new(rows) else {
            panic!("zero rows");
        };
        let Some(cols) = NonZeroUsize::new(cols) else {
            panic!("zero cols");
        };
        let mut matrix = Matrix::zero(rows, cols);
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let value = self.cell(Index { row, col });
                let row = match row.cmp(&index.row) {
                    std::cmp::Ordering::Less => row,
                    std::cmp::Ordering::Equal => continue,
                    std::cmp::Ordering::Greater => row - 1,
                };
                let col = match col.cmp(&index.col) {
                    std::cmp::Ordering::Less => col,
                    std::cmp::Ordering::Equal => continue,
                    std::cmp::Ordering::Greater => col - 1,
                };
                let index = Index { row, col };
                matrix.set_cell(index, value);
            }
        }
        matrix
    }

    pub fn inverse(&self) -> Matrix {
        let det = self.determinant();
        let mut matrix = self.adjugate();
        matrix.mul_scalar(1. / det);
        matrix
    }

    pub fn adjugate(&self) -> Matrix {
        self.matrix_of_cofactors().transpose()
    }

    pub fn matrix_of_minors(&self) -> Matrix {
        let mut matrix_of_minors = Matrix::zero(self.rows(), self.cols());
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let index = Index { row, col };
                let det = self.exclude_cross(index).determinant();
                matrix_of_minors.set_cell(index, det);
            }
        }
        matrix_of_minors
    }

    pub fn matrix_of_cofactors(&self) -> Matrix {
        let mut matrix = self.matrix_of_minors();
        for row in 0..matrix.rows().get() {
            for col in 0..matrix.cols().get() {
                let is_even = (row + col) % 2 == 0;
                let sign = if is_even { 1. } else { -1. };
                let index = Index { row, col };
                let value = matrix.cell(index);
                matrix.set_cell(index, value * sign);
            }
        }
        matrix
    }

    pub fn mul_matrix(&self, other: Self) -> Matrix {
        if self.cols() != other.rows() {
            panic!("unmatched matrix shapes for mul");
        }
        let mut matrix = Matrix::zero(self.rows(), other.cols());

        for row in 0..self.rows().get() {
            for col in 0..other.cols().get() {
                let index = Index { row, col };
                let mut sum = 0.;

                for i in 0..self.cols().get() {
                    let a = Index { row, col: i };
                    let b = Index { row: i, col };
                    let a = self.cell(a);
                    let b = other.cell(b);
                    sum += a * b;
                }

                matrix.set_cell(index, sum);
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let rows = NonZeroUsize::new(2).unwrap();
        let matrix = Matrix::new(rows, data);
        assert_eq!(matrix.cell(Index { row: 0, col: 0 }), 0.);
        assert_eq!(matrix.cell(Index { row: 0, col: 1 }), 1.);
        assert_eq!(matrix.cell(Index { row: 0, col: 2 }), 2.);
        assert_eq!(matrix.cell(Index { row: 1, col: 0 }), 3.);
        assert_eq!(matrix.cell(Index { row: 1, col: 1 }), 4.);
        assert_eq!(matrix.cell(Index { row: 1, col: 2 }), 5.);
    }

    #[test]
    fn test_add() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let rows = NonZeroUsize::new(2).unwrap();
        let mut matrix = Matrix::new(rows, data);
        matrix.add_scalar(1.);
        let expected = Matrix::new(rows, vec![1., 2., 3., 4., 5., 6.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_mul() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let rows = NonZeroUsize::new(2).unwrap();
        let mut matrix = Matrix::new(rows, data);
        matrix.mul_scalar(2.);
        let expected = Matrix::new(rows, vec![0., 2., 4., 6., 8., 10.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_transpose() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let rows = NonZeroUsize::new(2).unwrap();
        let matrix = Matrix::new(rows, data);
        let matrix = matrix.transpose();
        let expected = Matrix::new(
            NonZeroUsize::new(3).unwrap(),
            vec![
                0., 3., 1., //
                4., 2., 5., //
            ],
        );
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_determinant() {
        let data = vec![3., 8., 4., 6.];
        let rows = NonZeroUsize::new(2).unwrap();
        let matrix = Matrix::new(rows, data);
        assert_eq!(matrix.determinant(), -14.);

        let data = vec![
            6., 1., 1., //
            4., -2., 5., //
            2., 8., 7., //
        ];
        let rows = NonZeroUsize::new(3).unwrap();
        let matrix = Matrix::new(rows, data);
        assert_eq!(matrix.determinant(), -306.);
    }

    #[test]
    fn test_inverse() {
        let data = vec![
            3., 0., 2., //
            2., 0., -2., //
            0., 1., 1., //
        ];
        let rows = NonZeroUsize::new(3).unwrap();
        let matrix = Matrix::new(rows, data);
        let inverse = matrix.inverse();
        let expected = Matrix::new(
            rows,
            vec![
                0.2, 0.2, 0., //
                -0.2, 0.3, 1., //
                0.2, -0.3, 0., //
            ],
        );
        assert!(inverse.closes_to(&expected));

        let data = vec![
            2.0, 1.0, //
            1.0, 1.0, //
        ];
        let rows = NonZeroUsize::new(2).unwrap();
        let matrix = Matrix::new(rows, data);
        let inverse = matrix.inverse();
        let expected = Matrix::new(
            rows,
            vec![
                1.0, -1.0, //
                -1.0, 2.0, //
            ],
        );
        assert!(inverse.closes_to(&expected));
    }

    #[test]
    fn test_mul_matrix() {
        let data = vec![
            1., 2., 3., //
            4., 5., 6., //
        ];
        let rows = NonZeroUsize::new(2).unwrap();
        let a = Matrix::new(rows, data);

        let data = vec![
            7., 8., //
            9., 10., //
            11., 12., //
        ];
        let rows = NonZeroUsize::new(3).unwrap();
        let b = Matrix::new(rows, data);

        let matrix = a.mul_matrix(&b);
        let expected = Matrix::new(
            NonZeroUsize::new(2).unwrap(),
            vec![
                58., 64., //
                139., 154., //
            ],
        );
        assert!(matrix.closes_to(&expected));
    }
}
