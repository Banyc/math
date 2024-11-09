use core::{marker::PhantomData, num::NonZeroUsize};

use num_traits::{Float, One, Zero};
use primitive::{
    iter::Lookahead1,
    ops::{
        float::FloatExt,
        slice::{AsSlice, AsSliceMut},
    },
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub rows: NonZeroUsize,
    pub cols: NonZeroUsize,
}
impl Size {
    pub fn volume(&self) -> NonZeroUsize {
        let volume = self.rows.get() * self.cols.get();
        NonZeroUsize::new(volume).unwrap()
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

pub type ArrayMatrix<F, const N: usize> = MatrixBuf<[F; N], F>;
pub type VecMatrix<F> = MatrixBuf<Vec<F>, F>;
#[derive(Debug, Clone)]
pub struct MatrixBuf<T, F> {
    size: Size,
    buf: T,
    float: PhantomData<F>,
}
impl<T, F> Container2D<F> for MatrixBuf<T, F>
where
    T: AsSlice<F>,
    F: Float,
{
    fn size(&self) -> Size {
        self.size
    }
    fn get(&self, index: Index) -> F {
        let index = self.index_2_to_1(index);
        self.buf.as_slice()[index]
    }
}
impl<T, F> Container2DMut<F> for MatrixBuf<T, F>
where
    T: AsSliceMut<F>,
    F: Float,
{
    fn set(&mut self, index: Index, value: F) {
        let index = self.index_2_to_1(index);
        self.buf.as_slice_mut()[index] = value;
    }
}
impl<T, F> MatrixBuf<T, F>
where
    T: AsSlice<F>,
    F: Float,
{
    pub fn new(size: Size, buf: T) -> Self {
        if buf.as_slice().len() < size.volume().get() {
            panic!("not enough buffer size");
        }
        Self {
            size,
            buf,
            float: PhantomData,
        }
    }
    pub fn into_buffer(self) -> T {
        self.buf
    }
    pub fn buffer(&self) -> &T {
        &self.buf
    }
    pub fn buffer_mut(&mut self) -> &mut T {
        &mut self.buf
    }

    pub fn transpose(&self) -> Self
    where
        T: Clone + AsSliceMut<F>,
    {
        let buf = self.buf.clone();
        let size = Size {
            rows: self.size().cols,
            cols: self.size().rows,
        };
        let mut out = Self::new(size, buf);
        self.transpose_in(&mut out);
        out
    }
    pub fn determinant(&self) -> F {
        self.full_partial().determinant()
    }
    pub fn inverse(&self) -> VecMatrix<F> {
        self.full_partial().inverse()
    }
    pub fn mul_matrix(&self, other: &impl Container2D<F>) -> VecMatrix<F> {
        self.full_partial().mul_matrix(other)
    }
    pub fn submatrix(&self, excluded_rows: &[usize], excluded_cols: &[usize]) -> VecMatrix<F> {
        self.full_partial().submatrix(excluded_rows, excluded_cols)
    }

    fn full_partial(&self) -> PartialMatrix<'_, T, F> {
        let start = Index { row: 0, col: 0 };
        let end = Index {
            row: self.size().rows.get(),
            col: self.size().cols.get(),
        };
        PartialMatrix::new(self, start, end)
    }
    fn index_2_to_1(&self, index: Index) -> usize {
        if self.size().cols.get() <= index.col {
            panic!("col out of range");
        }
        if self.size().rows.get() <= index.row {
            panic!("row out of range");
        }
        index.to_1(self.size().cols)
    }
}
impl<F> VecMatrix<F>
where
    F: Float,
{
    pub fn zero(size: Size) -> Self {
        let buf = vec![Zero::zero(); size.volume().get()];
        MatrixBuf::new(size, buf)
    }
    pub fn identity(rows: NonZeroUsize) -> Self {
        let size = Size { rows, cols: rows };
        let mut matrix = Self::zero(size);
        identity(rows, &mut matrix);
        matrix
    }
}
impl<const N: usize, F> ArrayMatrix<F, N>
where
    F: Float,
{
    pub fn zero(size: Size) -> Self {
        let buf = [Zero::zero(); N];
        MatrixBuf::new(size, buf)
    }
    pub fn identity(rows: NonZeroUsize) -> Self {
        let size = Size { rows, cols: rows };
        let mut matrix = Self::zero(size);
        identity(rows, &mut matrix);
        matrix
    }

    pub fn mul_matrix_square(&self, other: &Self) -> Self {
        assert_eq!(self.size(), other.size());
        assert_eq!(self.size().rows, self.size().cols);
        let buf = [Zero::zero(); N];
        let mut out = Self::new(self.size(), buf);
        self.mul_matrix_in(other, &mut out);
        out
    }
}
fn identity<T>(rows: NonZeroUsize, zero: &mut impl Container2DMut<T>)
where
    T: One,
{
    for row in 0..rows.get() {
        let index = Index { row, col: row };
        zero.set(index, One::one());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PartialMatrix<'orig, T, F> {
    orig_matrix: &'orig MatrixBuf<T, F>,
    start: Index,
    /// exclusive
    end: Index,
}
impl<'orig, T, F> PartialMatrix<'orig, T, F>
where
    T: AsSlice<F>,
    F: Float,
{
    pub fn new(matrix: &'orig MatrixBuf<T, F>, start: Index, end: Index) -> Self {
        let start_in_bound =
            start.row < matrix.size().rows.get() && start.col < matrix.size().cols.get();
        let end_in_bound =
            end.row <= matrix.size().rows.get() && end.col <= matrix.size().cols.get();
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
impl<T, F> Container2D<F> for PartialMatrix<'_, T, F>
where
    T: AsSlice<F>,
    F: Float,
{
    fn size(&self) -> Size {
        let rows = self.end.row - self.start.row;
        let rows = NonZeroUsize::new(rows).unwrap();
        let cols = self.end.col - self.start.col;
        let cols = NonZeroUsize::new(cols).unwrap();
        Size { rows, cols }
    }
    fn get(&self, index: Index) -> F {
        let row = index.row + self.start.row;
        let col = index.col + self.start.col;
        let index = Index { row, col };
        self.orig_matrix.get(index)
    }
}
impl<T, F> PartialMatrix<'_, T, F>
where
    T: AsSlice<F>,
    F: Float,
{
    pub fn transpose(&self) -> VecMatrix<F> {
        let size = Size {
            rows: self.size().cols,
            cols: self.size().rows,
        };
        let mut matrix = VecMatrix::<F>::zero(size);
        self.transpose_in(&mut matrix);
        matrix
    }

    pub fn inverse(&self) -> VecMatrix<F> {
        let mut matrix_of_cofactors = VecMatrix::<F>::zero(self.matrix_of_cofactors_size());
        let mut submatrix = VecMatrix::<F>::zero(self.minor_size());
        let mut out = VecMatrix::<F>::zero(self.inverse_size());
        self.inverse_in(&mut matrix_of_cofactors, &mut submatrix, &mut out);
        out
    }

    pub fn mul_matrix(&self, other: &impl Container2D<F>) -> VecMatrix<F> {
        if self.size().cols != other.size().rows {
            panic!("unmatched matrix shapes for mul");
        }
        let size = Size {
            rows: self.size().rows,
            cols: other.size().cols,
        };
        let mut matrix = VecMatrix::<F>::zero(size);
        self.mul_matrix_in(other, &mut matrix);
        matrix
    }

    pub fn submatrix(&self, excluded_rows: &[usize], excluded_cols: &[usize]) -> VecMatrix<F> {
        let mut out =
            VecMatrix::<F>::zero(self.submatrix_size(excluded_rows.len(), excluded_cols.len()));
        self.submatrix_in(excluded_rows, excluded_cols, &mut out);
        out
    }

    pub fn collect(&self) -> VecMatrix<F> {
        let mut out = VecMatrix::zero(self.size());
        out.zip_mut_with(self, |_, x| x);
        out
    }
}

pub trait Container2D<T> {
    fn size(&self) -> Size;
    fn get(&self, index: Index) -> T;
}
pub trait Container2DMut<T>: Container2D<T> {
    fn set(&mut self, index: Index, value: T);
}
pub trait Matrix<T>: Container2D<T>
where
    T: Float,
{
    fn map_mut(&mut self, op: impl Fn(T) -> T)
    where
        Self: Container2DMut<T>,
    {
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let index = Index { row, col };
                let value = self.get(index);
                let value = op(value);
                self.set(index, value);
            }
        }
    }
    fn zip_mut_with(&mut self, other: &impl Container2D<T>, op: impl Fn(T, T) -> T)
    where
        Self: Container2DMut<T>,
    {
        assert_eq!(self.size(), other.size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let index = Index { row, col };
                let other = other.get(index);
                let this = self.get(index);
                let value = op(this, other);
                self.set(index, value);
            }
        }
    }
    fn closes_to(&self, other: &impl Container2D<T>) -> bool {
        assert_eq!(self.size(), other.size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let index = Index { row, col };
                let other = other.get(index);
                let this = self.get(index);
                if !this.closes_to(other) {
                    return false;
                }
            }
        }
        true
    }

    fn mul_matrix_size(&self, other: &impl Container2D<T>) -> Size {
        Size {
            rows: self.size().rows,
            cols: other.size().cols,
        }
    }
    fn mul_matrix_in(&self, other: &impl Container2D<T>, out: &mut impl Container2DMut<T>) {
        assert_eq!(out.size(), self.mul_matrix_size(other));
        for row in 0..self.size().rows.get() {
            for col in 0..other.size().cols.get() {
                let index = Index { row, col };
                let mut sum = Zero::zero();

                for i in 0..self.size().cols.get() {
                    let a = Index { row, col: i };
                    let b = Index { row: i, col };
                    let a = self.get(a);
                    let b = other.get(b);
                    sum = sum + (a * b);
                }

                out.set(index, sum);
            }
        }
    }

    fn transpose_size(&self) -> Size {
        Size {
            rows: self.size().cols,
            cols: self.size().rows,
        }
    }
    fn transpose_in(&self, out: &mut impl Container2DMut<T>) {
        assert_eq!(out.size(), self.transpose_size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let value = self.get(Index { row, col });
                let index = Index { row: col, col: row };
                out.set(index, value);
            }
        }
    }

    fn submatrix_size(&self, excluded_rows: usize, excluded_cols: usize) -> Size {
        let rows = self.size().rows.get().checked_sub(excluded_rows).unwrap();
        let cols = self.size().cols.get().checked_sub(excluded_cols).unwrap();
        let Some(rows) = NonZeroUsize::new(rows) else {
            panic!("zero rows");
        };
        let Some(cols) = NonZeroUsize::new(cols) else {
            panic!("zero cols");
        };
        Size { rows, cols }
    }
    fn submatrix_in(
        &self,
        excluded_rows: &[usize],
        excluded_cols: &[usize],
        out: &mut impl Container2DMut<T>,
    ) {
        assert_eq!(
            out.size(),
            self.submatrix_size(excluded_rows.len(), excluded_cols.len())
        );
        let excluded_rows_len = excluded_rows.len();
        let mut excluded_rows = Lookahead1::new(excluded_rows.iter().copied());
        let mut row_i = 0;
        for row in 0..self.size().rows.get() {
            if Some(row) == excluded_rows.peek().copied() {
                excluded_rows.pop();
                continue;
            }
            let excluded_cols_len = excluded_cols.len();
            let mut excluded_cols = Lookahead1::new(excluded_cols.iter().copied());
            let mut col_i = 0;
            for col in 0..self.size().cols.get() {
                if Some(col) == excluded_cols.peek().copied() {
                    excluded_cols.pop();
                    continue;
                }
                let value = self.get(Index { row, col });
                let index = Index {
                    row: row_i,
                    col: col_i,
                };
                out.set(index, value);
                col_i += 1;
            }
            assert_eq!(
                col_i + excluded_cols_len,
                self.size().cols.get(),
                "unordered `excluded_cols`"
            );
            row_i += 1;
        }
        assert_eq!(
            row_i + excluded_rows_len,
            self.size().rows.get(),
            "unordered `excluded_rows`"
        );
    }

    fn determinant(&self) -> T {
        if !self.size().is_square() {
            panic!("not a square matrix");
        }
        if self.size().rows.get() == 1 {
            return self.get(Index { row: 0, col: 0 });
        }
        if self.size().rows.get() == 2 {
            return self.get(Index { row: 0, col: 0 }) * self.get(Index { row: 1, col: 1 })
                - self.get(Index { row: 0, col: 1 }) * self.get(Index { row: 1, col: 0 });
        }

        let mut matrix = VecMatrix::<T>::zero(self.submatrix_size(1, 1));
        let mut sum = Zero::zero();
        let mut alt_sign = One::one();
        for col in 0..self.size().cols.get() {
            let index = Index { row: 0, col };
            let value = self.get(index);
            self.submatrix_in(&[0], &[col], &mut matrix);
            let det = matrix.determinant();
            sum = sum + (value * det * alt_sign);
            alt_sign = alt_sign.neg();
        }

        sum
    }

    fn matrix_of_minors_size(&self) -> Size {
        self.size()
    }
    fn minor_size(&self) -> Size {
        self.submatrix_size(1, 1)
    }
    fn matrix_of_minors_in(
        &self,
        minor: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        assert_eq!(out.size(), self.matrix_of_minors_size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let index = Index { row, col };
                self.submatrix_in(&[row], &[col], minor);
                let det = minor.determinant();
                out.set(index, det);
            }
        }
    }

    fn matrix_of_cofactors_size(&self) -> Size {
        self.matrix_of_minors_size()
    }
    fn matrix_of_cofactors_in(
        &self,
        minor: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        self.matrix_of_minors_in(minor, out);
        for row in 0..out.size().rows.get() {
            for col in 0..out.size().cols.get() {
                let is_even = (row + col) % 2 == 0;
                let sign = if is_even { 1. } else { -1. };
                let sign = T::from(sign).unwrap();
                let index = Index { row, col };
                let value = out.get(index);
                out.set(index, value * sign);
            }
        }
    }

    fn adjugate_size(&self) -> Size {
        self.transpose_size()
    }
    fn adjugate_in(
        &self,
        matrix_of_cofactors: &mut impl Container2DMut<T>,
        minor: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        self.matrix_of_cofactors_in(minor, matrix_of_cofactors);
        matrix_of_cofactors.transpose_in(out);
    }

    fn inverse_size(&self) -> Size {
        self.adjugate_size()
    }
    fn inverse_in(
        &self,
        matrix_of_cofactors: &mut impl Container2DMut<T>,
        minor: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        let det = self.determinant();
        self.adjugate_in(matrix_of_cofactors, minor, out);
        out.map_mut(|x| x / det);
    }
}
impl<T, F> Matrix<F> for T
where
    T: Container2D<F>,
    F: Float,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell() {
        let buf = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        assert_eq!(matrix.get(Index { row: 0, col: 0 }), 0.);
        assert_eq!(matrix.get(Index { row: 0, col: 1 }), 1.);
        assert_eq!(matrix.get(Index { row: 0, col: 2 }), 2.);
        assert_eq!(matrix.get(Index { row: 1, col: 0 }), 3.);
        assert_eq!(matrix.get(Index { row: 1, col: 1 }), 4.);
        assert_eq!(matrix.get(Index { row: 1, col: 2 }), 5.);
    }

    #[test]
    fn test_add() {
        let buf = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let mut matrix = MatrixBuf::new(size, buf);
        matrix.map_mut(|x| x + 1.);
        let expected = MatrixBuf::new(size, vec![1., 2., 3., 4., 5., 6.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_mul() {
        let buf = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let mut matrix = MatrixBuf::new(size, buf);
        matrix.map_mut(|x| x * 2.);
        let expected = MatrixBuf::new(size, vec![0., 2., 4., 6., 8., 10.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_transpose() {
        let buf = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        let matrix = matrix.transpose();
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let expected = MatrixBuf::new(
            size,
            vec![
                0., 3., 1., //
                4., 2., 5., //
            ],
        );
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_determinant() {
        let buf = vec![3., 8., 4., 6.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        assert_eq!(matrix.determinant(), -14.);

        let buf = vec![
            6., 1., 1., //
            4., -2., 5., //
            2., 8., 7., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        assert_eq!(matrix.determinant(), -306.);
    }

    #[test]
    fn test_inverse() {
        let buf = vec![
            3., 0., 2., //
            2., 0., -2., //
            0., 1., 1., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        let inverse = matrix.inverse();
        let expected = MatrixBuf::new(
            size,
            vec![
                0.2, 0.2, 0., //
                -0.2, 0.3, 1., //
                0.2, -0.3, 0., //
            ],
        );
        assert!(inverse.closes_to(&expected));

        let buf = vec![
            2.0, 1.0, //
            1.0, 1.0, //
        ];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let matrix = MatrixBuf::new(size, buf);
        let inverse = matrix.inverse();
        let expected = MatrixBuf::new(
            size,
            vec![
                1.0, -1.0, //
                -1.0, 2.0, //
            ],
        );
        assert!(inverse.closes_to(&expected));
    }

    #[test]
    fn test_mul_matrix() {
        let buf = vec![
            1., 2., 3., //
            4., 5., 6., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let a = MatrixBuf::new(size, buf);

        let buf = vec![
            7., 8., //
            9., 10., //
            11., 12., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let b = MatrixBuf::new(size, buf);

        let matrix = a.mul_matrix(&b);
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let expected = MatrixBuf::new(
            size,
            vec![
                58., 64., //
                139., 154., //
            ],
        );
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_vec_expand() {
        let mut v = vec![0; 1];
        v.push(1);
    }
}
