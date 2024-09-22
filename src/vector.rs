use std::{marker::PhantomData, num::NonZeroUsize};

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
    pub fn full(buf: T) -> Self {
        let size = NonZeroUsize::new(buf.as_slice().len()).unwrap();
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
impl<T, F> Container1D<F> for VectorBuf<T, F>
where
    T: Seq<F>,
{
    fn dims(&self) -> &[F] {
        &self.buf.as_slice()[..self.size.get()]
    }
}
impl<T, F> Container1DMut<F> for VectorBuf<T, F>
where
    T: SeqMut<F>,
{
    fn dims_mut(&mut self) -> &mut [F] {
        &mut self.buf.as_slice_mut()[..self.size.get()]
    }
}

pub trait Container1D<F> {
    #[must_use]
    fn dims(&self) -> &[F];
}
pub trait Container1DMut<F>: Container1D<F> {
    #[must_use]
    fn dims_mut(&mut self) -> &mut [F];
}
impl<T, F> Vector<F> for T
where
    T: Container1DMut<F>,
    F: Float + core::fmt::Debug + core::iter::Sum,
{
}
pub trait Vector<F>: Container1DMut<F>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    fn zip_mut_with<T>(&mut self, other: &T, op: impl Fn(&F, &F) -> F)
    where
        T: Container1D<F>,
        F: core::fmt::Debug,
    {
        assert_eq!(self.dims().len(), other.dims().len());
        self.dims_mut()
            .iter_mut()
            .zip(other.dims().iter())
            .for_each(|(a, b)| {
                let res = op(a, b);
                *a = res;
            });
    }
    fn add(&mut self, other: &impl Container1D<F>) {
        self.zip_mut_with(other, |&a, &b| a + b)
    }
    fn sub(&mut self, other: &impl Container1D<F>) {
        self.zip_mut_with(other, |&a, &b| a - b)
    }
    #[must_use]
    fn dot(&mut self, other: &impl Container1D<F>) -> F {
        self.zip_mut_with(other, |&a, &b| a * b);
        self.dims().iter().copied().sum::<F>()
    }
    fn mul(&mut self, other: F) {
        self.dims_mut().iter_mut().for_each(|x| *x = *x * other);
    }
    fn div(&mut self, other: F) {
        self.mul(F::one() / other);
    }
    #[must_use]
    fn mag(&self) -> F {
        let sum = self.dims().iter().map(|x| x.powi(2)).sum::<F>();
        sum.sqrt()
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
    fn dist(&mut self, other: &impl Container1D<F>) -> F {
        self.sub(other);
        self.mag()
    }
    fn lerp(&mut self, other: &impl Container1D<F>, t: NormalizedF64) {
        self.zip_mut_with(other, |&a, &b| lerp(&(a..=b), t.into()));
    }
    #[must_use]
    fn heading_angle(&self, adjacent_axis: usize, opposite_axis: usize) -> F {
        let adj = self.dims()[adjacent_axis];
        let opp = self.dims()[opposite_axis];
        F::atan2(opp, adj)
    }
    fn rotate(&mut self, adjacent_axis: usize, opposite_axis: usize, angle: F) {
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.dims()[adjacent_axis];
        let y = self.dims()[opposite_axis];
        let x_ = cos * x + -sin * y;
        let y_ = sin * x + cos * y;
        self.dims_mut()[adjacent_axis] = x_;
        self.dims_mut()[opposite_axis] = y_;
    }
    #[must_use]
    fn angle_between(&mut self, other: &impl Vector<F>) -> F {
        let mag = self.mag();
        let dot = self.dot(other);
        let mul_mag = mag * other.mag();
        F::acos(dot / mul_mag)
    }
    #[must_use]
    fn normal_point<E>(&self, start: &impl Vector<F>, end: &E) -> E
    where
        Self: Clone,
        E: Vector<F> + Clone,
    {
        assert_eq!(self.dims().len(), start.dims().len());
        assert_eq!(start.dims().len(), end.dims().len());
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
    #[must_use]
    fn heading_angle_2d(&self) -> F {
        assert_eq!(self.dims().len(), 2);
        self.heading_angle(0, 1)
    }
    fn rotate_2d(&mut self, angle: F) {
        assert_eq!(self.dims().len(), 2);
        self.rotate(0, 1, angle)
    }
    /// The cross product is only defined in 3D space and takes two non-parallel vectors as input and produces a third vector that is orthogonal to both the input vectors.
    /// - ref: <https://learnopengl.com/Getting-started/Transformations>
    #[must_use]
    fn cross(&self, other: &impl Container1D<F>) -> ArrayVector<F, 3> {
        assert_eq!(self.dims().len(), 3);
        assert_eq!(self.dims().len(), other.dims().len());
        let dims = [
            self.dims()[1] * other.dims()[2] - self.dims()[2] * other.dims()[1],
            self.dims()[2] * other.dims()[0] - self.dims()[0] * other.dims()[2],
            self.dims()[0] * other.dims()[1] - self.dims()[1] * other.dims()[0],
        ];
        ArrayVector::full(dims)
    }
}
impl<T, F> From<VectorBuf<T, F>> for matrix::MatrixBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    fn from(value: VectorBuf<T, F>) -> Self {
        let size = matrix::Size {
            rows: NonZeroUsize::new(value.dims().len()).unwrap(),
            cols: NonZeroUsize::new(1).unwrap(),
        };
        matrix::MatrixBuf::new(size, value.into_buffer())
    }
}
impl<T, F> From<matrix::MatrixBuf<T, F>> for VectorBuf<T, F>
where
    T: Seq<F>,
    F: Float,
{
    fn from(value: matrix::MatrixBuf<T, F>) -> Self {
        assert_eq!(value.size().cols.get(), 1);
        let size = value.size().rows;
        Self::new(size, value.into_buffer())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use primitive::float::FloatExt;

    use super::*;

    #[test]
    fn test_add() {
        let mut a = ArrayVector::full([0., 1.]);
        let b = ArrayVector::full([1., 2.]);
        a.add(&b);
        assert_eq!(a.dims(), &[1., 3.]);
    }

    #[test]
    fn test_heading() {
        let v = ArrayVector::full([1., 1.]);
        assert!(v.heading_angle_2d().closes_to(PI / 4.));
        let v = ArrayVector::full([-1., 1.]);
        assert!(v.heading_angle_2d().closes_to(PI / 2. + PI / 4.));
        let v = ArrayVector::full([-1., -1.]);
        assert!(v.heading_angle_2d().closes_to(PI + PI / 4. - 2. * PI));
        let v = ArrayVector::full([1., -1.]);
        assert!(v.heading_angle_2d().closes_to(-PI / 4.));
    }

    #[test]
    fn test_rotation() {
        let mut v = ArrayVector::full([1., 0.5]);
        v.rotate_2d(PI / 2.);
        assert!(v.dims()[0].closes_to(-0.5));
        assert!(v.dims()[1].closes_to(1.));
    }
}
