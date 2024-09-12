use std::num::NonZeroUsize;

use num_traits::Float;
use primitive::{float::FloatExt, seq::Seq};

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
pub struct Matrix<T> {
    rows: NonZeroUsize,
    data: T,
}
impl<T> Container2D for Matrix<T>
where
    T: Seq<f64>,
{
    type Item = f64;
    fn rows(&self) -> NonZeroUsize {
        self.rows
    }
    fn cols(&self) -> NonZeroUsize {
        let cols = self.data.as_slice().len() / self.rows.get();
        NonZeroUsize::new(cols).unwrap()
    }
    fn cell(&self, index: Index) -> Self::Item {
        let index = self.index_2_to_1(index);
        self.data.as_slice()[index]
    }
}
impl<T> Container2DMut for Matrix<T>
where
    T: Seq<f64>,
{
    fn set_cell(&mut self, index: Index, value: Self::Item) {
        let index = self.index_2_to_1(index);
        self.data.as_slice_mut()[index] = value;
    }
}
impl<T> Matrix<T>
where
    T: Seq<f64>,
{
    pub fn new(rows: NonZeroUsize, data: T) -> Self {
        if data.as_slice().len() % rows.get() != 0 {
            panic!("not even");
        }
        if data.as_slice().is_empty() {
            panic!("no data");
        }
        Self { rows, data }
    }
    pub fn zero(rows: NonZeroUsize, cols: NonZeroUsize) -> Matrix<Vec<f64>> {
        let data = vec![0.; rows.get() * cols.get()];
        Matrix::new(rows, data)
    }
    pub fn identity(rows: NonZeroUsize) -> Matrix<Vec<f64>> {
        let mut matrix = Self::zero(rows, rows);
        for row in 0..rows.get() {
            let index = Index { row, col: row };
            matrix.set_cell(index, 1.);
        }
        matrix
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
        for cell in self.data.as_slice_mut() {
            *cell += value;
        }
    }
    pub fn mul_scalar(&mut self, value: f64) {
        for cell in self.data.as_slice_mut() {
            *cell *= value;
        }
    }

    pub fn add_matrix(&mut self, other: &impl Container2D<Item = f64>) {
        self.assert_same_shape(other);
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let index = Index { row, col };
                let value = other.cell(index);
                self.set_cell(index, value);
            }
        }
    }
    pub fn assert_same_shape(&self, other: &impl Container2D<Item = f64>) {
        if self.rows() != other.rows() {
            panic!("unmatched rows");
        }
        if self.cols() != other.cols() {
            panic!("unmatched cols");
        }
    }

    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        let data = self.data.clone();
        let mut out = Self::new(self.cols(), data);
        transpose(self, &mut out);
        out
    }
    pub fn determinant(&self) -> f64 {
        self.full_partial().determinant()
    }
    pub fn inverse(&self) -> Matrix<Vec<f64>> {
        self.full_partial().inverse()
    }

    fn full_partial(&self) -> PartialMatrix<'_, T> {
        let start = Index { row: 0, col: 0 };
        let end = Index {
            row: self.rows().get(),
            col: self.cols().get(),
        };
        PartialMatrix::new(self, start, end)
    }

    pub fn closes_to(&self, other: &impl Container2D<Item = f64>) -> bool {
        self.assert_same_shape(other);
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let index = Index { row, col };
                let other = other.cell(index);
                let this = self.cell(index);
                if !this.closes_to(other) {
                    return false;
                }
            }
        }
        true
    }

    pub fn mul_matrix(&self, other: &Self) -> Matrix<Vec<f64>> {
        self.full_partial().mul_matrix(other.full_partial())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PartialMatrix<'orig, T> {
    orig_matrix: &'orig Matrix<T>,
    start: Index,
    /// exclusive
    end: Index,
}
impl<'orig, T> PartialMatrix<'orig, T> {
    pub fn new(matrix: &'orig Matrix<T>, start: Index, end: Index) -> Self
    where
        T: Seq<f64>,
    {
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
impl<T> Container2D for PartialMatrix<'_, T>
where
    T: Seq<f64>,
{
    type Item = f64;
    fn rows(&self) -> NonZeroUsize {
        let rows = self.end.row - self.start.row;
        NonZeroUsize::new(rows).unwrap()
    }
    fn cols(&self) -> NonZeroUsize {
        let cols = self.end.col - self.start.col;
        NonZeroUsize::new(cols).unwrap()
    }
    fn cell(&self, index: Index) -> Self::Item {
        let row = index.row + self.start.row;
        let col = index.col + self.start.col;
        let index = Index { row, col };
        self.orig_matrix.cell(index)
    }
}
impl<T> PartialMatrix<'_, T>
where
    T: Seq<f64>,
{
    pub fn transpose(&self) -> Matrix<Vec<f64>> {
        let mut matrix = Matrix::<Vec<f64>>::zero(self.cols(), self.rows());
        transpose(self, &mut matrix);
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

    pub fn exclude_cross(&self, index: Index) -> Matrix<Vec<f64>> {
        let _valid = self.cell(index);
        let rows = self.rows().get() - 1;
        let cols = self.cols().get() - 1;
        let Some(rows) = NonZeroUsize::new(rows) else {
            panic!("zero rows");
        };
        let Some(cols) = NonZeroUsize::new(cols) else {
            panic!("zero cols");
        };
        let mut matrix = Matrix::<Vec<f64>>::zero(rows, cols);
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

    pub fn inverse(&self) -> Matrix<Vec<f64>> {
        let det = self.determinant();
        let mut matrix = self.adjugate();
        matrix.mul_scalar(1. / det);
        matrix
    }

    pub fn adjugate(&self) -> Matrix<Vec<f64>> {
        self.matrix_of_cofactors().transpose()
    }

    pub fn matrix_of_minors(&self) -> Matrix<Vec<f64>> {
        let mut matrix_of_minors = Matrix::<Vec<f64>>::zero(self.rows(), self.cols());
        for row in 0..self.rows().get() {
            for col in 0..self.cols().get() {
                let index = Index { row, col };
                let det = self.exclude_cross(index).determinant();
                matrix_of_minors.set_cell(index, det);
            }
        }
        matrix_of_minors
    }

    pub fn matrix_of_cofactors(&self) -> Matrix<Vec<f64>> {
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

    pub fn mul_matrix(&self, other: Self) -> Matrix<Vec<f64>> {
        if self.cols() != other.rows() {
            panic!("unmatched matrix shapes for mul");
        }
        let mut matrix = Matrix::<Vec<f64>>::zero(self.rows(), other.cols());
        mul_matrix(self, &other, &mut matrix);
        matrix
    }
}

pub trait Container2D {
    type Item;
    fn rows(&self) -> NonZeroUsize;
    fn cols(&self) -> NonZeroUsize;
    fn cell(&self, index: Index) -> Self::Item;

    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }
}
pub trait Container2DMut: Container2D {
    fn set_cell(&mut self, index: Index, value: Self::Item);
}
fn mul_matrix<T: Float>(
    this: &impl Container2D<Item = T>,
    other: &impl Container2D<Item = T>,
    out: &mut impl Container2DMut<Item = T>,
) {
    assert_eq!(out.rows(), this.rows());
    assert_eq!(out.cols(), other.cols());
    for row in 0..this.rows().get() {
        for col in 0..other.cols().get() {
            let index = Index { row, col };
            let mut sum = T::zero();

            for i in 0..this.cols().get() {
                let a = Index { row, col: i };
                let b = Index { row: i, col };
                let a = this.cell(a);
                let b = other.cell(b);
                sum = sum + (a * b);
            }

            out.set_cell(index, sum);
        }
    }
}
fn transpose<T: Float>(this: &impl Container2D<Item = T>, out: &mut impl Container2DMut<Item = T>) {
    assert_eq!(out.rows(), this.cols());
    assert_eq!(out.cols(), this.rows());
    for row in 0..this.rows().get() {
        for col in 0..this.cols().get() {
            let value = this.cell(Index { row, col });
            let index = Index { row: col, col: row };
            out.set_cell(index, value);
        }
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
