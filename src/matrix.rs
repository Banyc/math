use std::{marker::PhantomData, num::NonZeroUsize};

use num_traits::{Float, One, Zero};
use primitive::{
    float::FloatExt,
    seq::{Seq, SeqMut},
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

pub type ArrayMatrixBuf<F, const N: usize> = MatrixBuf<[F; N], F>;
pub type VecMatrixBuf<F> = MatrixBuf<Vec<F>, F>;
#[derive(Debug, Clone)]
pub struct MatrixBuf<T, F> {
    size: Size,
    data: T,
    float: PhantomData<F>,
}
impl<T, F> Container2D<F> for MatrixBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    fn size(&self) -> Size {
        self.size
    }
    fn get(&self, index: Index) -> F {
        let index = self.index_2_to_1(index);
        self.data.as_slice()[index]
    }
}
impl<T, F> Container2DMut<F> for MatrixBuf<T, F>
where
    T: SeqMut<F>,
    F: Float,
{
    fn set(&mut self, index: Index, value: F) {
        let index = self.index_2_to_1(index);
        self.data.as_slice_mut()[index] = value;
    }
}
impl<T, F> Seq<F> for MatrixBuf<T, F>
where
    T: SeqMut<F>,
    F: Float,
{
    fn as_slice(&self) -> &[F] {
        self.data.as_slice()
    }
}
impl<T, F> SeqMut<F> for MatrixBuf<T, F>
where
    T: SeqMut<F>,
    F: Float,
{
    fn as_slice_mut(&mut self) -> &mut [F] {
        self.data.as_slice_mut()
    }
}
impl<T, F> MatrixBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    pub fn new(size: Size, data: T) -> Self {
        if data.as_slice().len() < size.volume().get() {
            panic!("not enough buffer size");
        }
        Self {
            size,
            data,
            float: PhantomData,
        }
    }
    pub fn into_buffer(self) -> T {
        self.data
    }
    pub fn zero(size: Size) -> VecMatrixBuf<F> {
        let data = vec![Zero::zero(); size.volume().get()];
        MatrixBuf::new(size, data)
    }
    pub fn identity(rows: NonZeroUsize) -> VecMatrixBuf<F> {
        let size = Size { rows, cols: rows };
        let mut matrix = Self::zero(size);
        for row in 0..rows.get() {
            let index = Index { row, col: row };
            matrix.set(index, One::one());
        }
        matrix
    }

    pub fn transpose(&self) -> Self
    where
        T: Clone + SeqMut<F>,
    {
        let data = self.data.clone();
        let size = Size {
            rows: self.size().cols,
            cols: self.size().rows,
        };
        let mut out = Self::new(size, data);
        self.transpose_in(&mut out);
        out
    }
    pub fn determinant(&self) -> F {
        self.full_partial().determinant()
    }
    pub fn inverse(&self) -> VecMatrixBuf<F> {
        self.full_partial().inverse()
    }
    pub fn mul_matrix(&self, other: &impl Container2D<F>) -> VecMatrixBuf<F> {
        self.full_partial().mul_matrix(other)
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

#[derive(Debug, Clone, Copy)]
pub struct PartialMatrix<'orig, T, F> {
    orig_matrix: &'orig MatrixBuf<T, F>,
    start: Index,
    /// exclusive
    end: Index,
}
impl<'orig, T, F> PartialMatrix<'orig, T, F>
where
    T: Seq<F>,
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
    T: Seq<F>,
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
    T: Seq<F>,
    F: Float,
{
    pub fn transpose(&self) -> VecMatrixBuf<F> {
        let size = Size {
            rows: self.size().cols,
            cols: self.size().rows,
        };
        let mut matrix = VecMatrixBuf::<F>::zero(size);
        self.transpose_in(&mut matrix);
        matrix
    }

    pub fn inverse(&self) -> VecMatrixBuf<F> {
        let mut matrix_of_cofactors = VecMatrixBuf::<F>::zero(self.matrix_of_cofactors_size());
        let mut out = VecMatrixBuf::<F>::zero(self.inverse_size());
        self.inverse_in(&mut matrix_of_cofactors, &mut out);
        out
    }

    pub fn mul_matrix(&self, other: &impl Container2D<F>) -> VecMatrixBuf<F> {
        if self.size().cols != other.size().rows {
            panic!("unmatched matrix shapes for mul");
        }
        let size = Size {
            rows: self.size().rows,
            cols: other.size().cols,
        };
        let mut matrix = VecMatrixBuf::<F>::zero(size);
        self.mul_matrix_in(other, &mut matrix);
        matrix
    }

    pub fn collect(&self) -> VecMatrixBuf<F> {
        let mut out = VecMatrixBuf::zero(self.size());
        out.zip_mut(self, |_, x| x);
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
    fn for_each(&mut self, op: impl Fn(T) -> T)
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
    fn zip_mut(&mut self, other: &impl Container2D<T>, op: impl Fn(T, T) -> T)
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

    fn exclude_cross_size(&self) -> Size {
        let rows = self.size().rows.get() - 1;
        let cols = self.size().cols.get() - 1;
        let Some(rows) = NonZeroUsize::new(rows) else {
            panic!("zero rows");
        };
        let Some(cols) = NonZeroUsize::new(cols) else {
            panic!("zero cols");
        };
        Size { rows, cols }
    }
    fn exclude_cross_in(&self, index: Index, out: &mut impl Container2DMut<T>) {
        let _valid = self.get(index);
        assert_eq!(out.size(), self.exclude_cross_size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let value = self.get(Index { row, col });
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
                out.set(index, value);
            }
        }
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

        let mut matrix = VecMatrixBuf::<T>::zero(self.exclude_cross_size());
        let mut sum = Zero::zero();
        let mut alt_sign = One::one();
        for col in 0..self.size().cols.get() {
            let index = Index { row: 0, col };
            let value = self.get(index);
            self.exclude_cross_in(index, &mut matrix);
            let det = matrix.determinant();
            sum = sum + (value * det * alt_sign);
            alt_sign = alt_sign.neg();
        }

        sum
    }

    fn matrix_of_minors_size(&self) -> Size {
        self.size()
    }
    fn matrix_of_minors_in(
        &self,
        exclude_cross: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        assert_eq!(out.size(), self.matrix_of_minors_size());
        for row in 0..self.size().rows.get() {
            for col in 0..self.size().cols.get() {
                let index = Index { row, col };
                self.exclude_cross_in(index, exclude_cross);
                let det = exclude_cross.determinant();
                out.set(index, det);
            }
        }
    }

    fn matrix_of_cofactors_size(&self) -> Size {
        self.matrix_of_minors_size()
    }
    fn matrix_of_cofactors_in(
        &self,
        exclude_cross: &mut impl Container2DMut<T>,
        out: &mut impl Container2DMut<T>,
    ) {
        self.matrix_of_minors_in(exclude_cross, out);
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
    fn adjugate_in<O>(&self, matrix_of_cofactors: &mut impl Container2DMut<T>, out: &mut O)
    where
        O: Container2DMut<T> + SeqMut<T>,
    {
        let mut exclude_cross = MatrixBuf::new(self.exclude_cross_size(), out.as_slice_mut());
        self.matrix_of_cofactors_in(&mut exclude_cross, matrix_of_cofactors);
        matrix_of_cofactors.transpose_in(out);
    }

    fn inverse_size(&self) -> Size {
        self.adjugate_size()
    }
    fn inverse_in<O>(&self, matrix_of_cofactors: &mut impl Container2DMut<T>, out: &mut O)
    where
        O: Container2DMut<T> + SeqMut<T>,
    {
        let det = self.determinant();
        self.adjugate_in(matrix_of_cofactors, out);
        out.for_each(|x| x / det);
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
        let data = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
        assert_eq!(matrix.get(Index { row: 0, col: 0 }), 0.);
        assert_eq!(matrix.get(Index { row: 0, col: 1 }), 1.);
        assert_eq!(matrix.get(Index { row: 0, col: 2 }), 2.);
        assert_eq!(matrix.get(Index { row: 1, col: 0 }), 3.);
        assert_eq!(matrix.get(Index { row: 1, col: 1 }), 4.);
        assert_eq!(matrix.get(Index { row: 1, col: 2 }), 5.);
    }

    #[test]
    fn test_add() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let mut matrix = MatrixBuf::new(size, data);
        matrix.for_each(|x| x + 1.);
        let expected = MatrixBuf::new(size, vec![1., 2., 3., 4., 5., 6.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_mul() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let mut matrix = MatrixBuf::new(size, data);
        matrix.for_each(|x| x * 2.);
        let expected = MatrixBuf::new(size, vec![0., 2., 4., 6., 8., 10.]);
        assert!(matrix.closes_to(&expected));
    }

    #[test]
    fn test_transpose() {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
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
        let data = vec![3., 8., 4., 6.];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
        assert_eq!(matrix.determinant(), -14.);

        let data = vec![
            6., 1., 1., //
            4., -2., 5., //
            2., 8., 7., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
        assert_eq!(matrix.determinant(), -306.);
    }

    #[test]
    fn test_inverse() {
        let data = vec![
            3., 0., 2., //
            2., 0., -2., //
            0., 1., 1., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
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

        let data = vec![
            2.0, 1.0, //
            1.0, 1.0, //
        ];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let matrix = MatrixBuf::new(size, data);
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
        let data = vec![
            1., 2., 3., //
            4., 5., 6., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(2).unwrap(),
            cols: NonZeroUsize::new(3).unwrap(),
        };
        let a = MatrixBuf::new(size, data);

        let data = vec![
            7., 8., //
            9., 10., //
            11., 12., //
        ];
        let size = Size {
            rows: NonZeroUsize::new(3).unwrap(),
            cols: NonZeroUsize::new(2).unwrap(),
        };
        let b = MatrixBuf::new(size, data);

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
}
