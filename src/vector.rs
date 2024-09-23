use core::{marker::PhantomData, num::NonZeroUsize};

use num_traits::Float;
use primitive::seq::{Seq, SeqMut};
use strict_num::NormalizedF64;

use crate::{
    graphics::lerp,
    matrix::{self, Container2D},
};

pub type VecVector<F> = VectorBuf<Vec<F>, F>;
pub type ArrayVector<F, const N: usize> = VectorBuf<[F; N], F>;
#[derive(Debug, Clone, Copy)]
pub struct VectorBuf<T, F> {
    size: NonZeroUsize,
    buf: T,
    float: PhantomData<F>,
}
impl<T, F> VectorBuf<T, F>
where
    T: Seq<F>,
{
    pub fn new(size: NonZeroUsize, buf: T) -> Self {
        assert!(size.get() <= buf.as_slice().len());
        Self {
            size,
            buf,
            float: PhantomData,
        }
    }
    pub fn into_buffer(self) -> T {
        self.buf
    }
}
impl<T, F> Seq<F> for VectorBuf<T, F>
where
    T: Seq<F>,
{
    fn as_slice(&self) -> &[F] {
        &self.buf.as_slice()[..self.size.get()]
    }
}
impl<T, F> SeqMut<F> for VectorBuf<T, F>
where
    T: SeqMut<F>,
{
    fn as_slice_mut(&mut self) -> &mut [F] {
        &mut self.buf.as_slice_mut()[..self.size.get()]
    }
}

pub trait Vector<F>: Seq<F>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    fn closes_to(&self, other: &impl Seq<F>) -> bool {
        use primitive::float::FloatExt;
        assert_eq!(self.as_slice().len(), other.as_slice().len());
        self.as_slice()
            .iter()
            .copied()
            .zip(other.as_slice().iter().copied())
            .all(|(a, b)| a.closes_to(b))
    }
    #[must_use]
    fn mag(&self) -> F {
        let sum = self.as_slice().iter().map(|x| x.powi(2)).sum::<F>();
        sum.sqrt()
    }
    #[must_use]
    fn heading_angle(&self, adjacent_axis: usize, opposite_axis: usize) -> F {
        let adj = self.as_slice()[adjacent_axis];
        let opp = self.as_slice()[opposite_axis];
        F::atan2(opp, adj)
    }
    #[must_use]
    fn heading_angle_2d(&self) -> F {
        assert_eq!(self.as_slice().len(), 2);
        self.heading_angle(0, 1)
    }
    /// The cross product is only defined in 3D space and takes two non-parallel vectors as input and produces a third vector that is orthogonal to both the input vectors.
    /// - ref: <https://learnopengl.com/Getting-started/Transformations>
    #[must_use]
    fn cross(&self, other: &impl Seq<F>) -> [F; 3] {
        assert_eq!(self.as_slice().len(), 3);
        assert_eq!(self.as_slice().len(), other.as_slice().len());
        [
            self.as_slice()[1] * other.as_slice()[2] - self.as_slice()[2] * other.as_slice()[1],
            self.as_slice()[2] * other.as_slice()[0] - self.as_slice()[0] * other.as_slice()[2],
            self.as_slice()[0] * other.as_slice()[1] - self.as_slice()[1] * other.as_slice()[0],
        ]
    }
    fn into_matrix(self) -> Option<matrix::MatrixBuf<Self, F>>
    where
        Self: Sized,
    {
        let rows = NonZeroUsize::new(self.as_slice().len())?;
        let size = matrix::Size {
            rows,
            cols: NonZeroUsize::new(1).unwrap(),
        };
        Some(matrix::MatrixBuf::new(size, self))
    }
}
impl<T, F> Vector<F> for T
where
    T: Seq<F>,
    F: Float + core::fmt::Debug + core::iter::Sum,
{
}

pub trait VectorMut<F>: SeqMut<F> + Vector<F>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    fn zip_mut_with<T>(&mut self, other: &T, op: impl Fn(&F, &F) -> F)
    where
        T: Seq<F>,
        F: core::fmt::Debug,
    {
        assert_eq!(self.as_slice().len(), other.as_slice().len());
        self.as_slice_mut()
            .iter_mut()
            .zip(other.as_slice().iter())
            .for_each(|(a, b)| {
                let res = op(a, b);
                *a = res;
            });
    }
    fn add(&mut self, other: &impl Seq<F>) {
        self.zip_mut_with(other, |&a, &b| a + b)
    }
    fn sub(&mut self, other: &impl Seq<F>) {
        self.zip_mut_with(other, |&a, &b| a - b)
    }
    #[must_use]
    fn dot(&mut self, other: &impl Seq<F>) -> F {
        self.zip_mut_with(other, |&a, &b| a * b);
        self.as_slice().iter().copied().sum::<F>()
    }
    fn mul(&mut self, other: F) {
        self.as_slice_mut().iter_mut().for_each(|x| *x = *x * other);
    }
    fn div(&mut self, other: F) {
        self.mul(F::one() / other);
    }
    fn normalize(&mut self) {
        self.div(self.mag());
    }
    fn set_mag(&mut self, mag: F) {
        self.normalize();
        self.mul(mag);
    }
    fn limit(&mut self, min: F, max: F) {
        let mag = self.mag().clamp(min, max);
        self.set_mag(mag);
    }
    #[must_use]
    fn dist(&mut self, other: &impl Seq<F>) -> F {
        self.sub(other);
        self.mag()
    }
    fn lerp(&mut self, other: &impl Seq<F>, t: NormalizedF64) {
        self.zip_mut_with(other, |&a, &b| lerp(&(a..=b), t.into()));
    }
    fn rotate(&mut self, adjacent_axis: usize, opposite_axis: usize, angle: F) {
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.as_slice()[adjacent_axis];
        let y = self.as_slice()[opposite_axis];
        let x_ = cos * x + -sin * y;
        let y_ = sin * x + cos * y;
        self.as_slice_mut()[adjacent_axis] = x_;
        self.as_slice_mut()[opposite_axis] = y_;
    }
    #[must_use]
    fn angle_between(&mut self, other: &impl Vector<F>) -> F {
        let mag = self.mag();
        let dot = self.dot(other);
        let mul_mag = mag * other.mag();
        F::acos(dot / mul_mag)
    }
    fn rotate_2d(&mut self, angle: F) {
        assert_eq!(self.as_slice().len(), 2);
        self.rotate(0, 1, angle)
    }
    #[must_use]
    fn normal_point<E>(&self, start: &impl Vector<F>, end: &E) -> E
    where
        Self: Clone,
        E: VectorMut<F> + Clone,
    {
        assert_eq!(self.as_slice().len(), start.as_slice().len());
        assert_eq!(start.as_slice().len(), end.as_slice().len());
        let mut a = self.clone();
        a.sub(start);
        let mag = a.mag();
        let mut b = end.clone();
        b.sub(start);
        let angle = a.angle_between(&b);
        drop(a);
        let d = mag * F::cos(angle);
        b.set_mag(d);
        b
    }
}
impl<T, F> VectorMut<F> for T
where
    T: SeqMut<F>,
    F: Float + core::fmt::Debug + core::iter::Sum,
{
}
impl<T, F> From<VectorBuf<T, F>> for matrix::MatrixBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    fn from(value: VectorBuf<T, F>) -> Self {
        let size = matrix::Size {
            rows: NonZeroUsize::new(value.as_slice().len()).unwrap(),
            cols: NonZeroUsize::new(1).unwrap(),
        };
        matrix::MatrixBuf::new(size, value.into_buffer())
    }
}
impl<T, F> TryFrom<matrix::MatrixBuf<T, F>> for VectorBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    type Error = &'static str;
    fn try_from(value: matrix::MatrixBuf<T, F>) -> Result<Self, Self::Error> {
        if value.size().cols.get() != 1 {
            return Err("multiple columns");
        }
        let size = value.size().rows;
        Ok(Self::new(size, value.into_buffer()))
    }
}
impl<F, const N: usize> TryFrom<matrix::ArrayMatrix<F, N>> for [F; N]
where
    F: Float,
{
    type Error = &'static str;
    fn try_from(value: matrix::ArrayMatrix<F, N>) -> Result<Self, Self::Error> {
        if value.size().cols.get() != 1 {
            return Err("multiple columns");
        }
        let size = value.size().rows;
        let buf = value.into_buffer();
        assert_eq!(size.get(), buf.len());
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use primitive::float::FloatExt;

    use super::*;

    #[test]
    fn test_add() {
        let mut a = [0., 1.];
        let b = [1., 2.];
        a.add(&b);
        assert_eq!(a.as_slice(), &[1., 3.]);
    }

    #[test]
    fn test_heading() {
        let v = [1., 1.];
        assert!(v.heading_angle_2d().closes_to(PI / 4.));
        let v = [-1., 1.];
        assert!(v.heading_angle_2d().closes_to(PI / 2. + PI / 4.));
        let v = [-1., -1.];
        assert!(v.heading_angle_2d().closes_to(PI + PI / 4. - 2. * PI));
        let v = [1., -1.];
        assert!(v.heading_angle_2d().closes_to(-PI / 4.));
    }

    #[test]
    fn test_rotation() {
        let mut v = [1., 0.5];
        v.rotate_2d(PI / 2.);
        assert!(v.closes_to(&[-0.5, 1.]));
    }
}
