use std::num::NonZeroUsize;

use num_traits::Float;
use strict_num::NormalizedF64;

use crate::{
    graphics::lerp,
    matrix::{self, Container2D},
};

pub use minimal::*;
mod minimal {
    #[derive(Debug, Clone, Copy)]
    pub struct Vector<F, const N: usize> {
        dims: [F; N],
    }
    impl<F, const N: usize> Vector<F, N> {
        #[must_use]
        pub fn new(dims: [F; N]) -> Self {
            Self { dims }
        }
        #[must_use]
        pub fn dims(&self) -> &[F; N] {
            &self.dims
        }
        #[must_use]
        pub fn dims_mut(&mut self) -> &mut [F; N] {
            &mut self.dims
        }
        #[must_use]
        pub fn zip_with(&self, other: &Self, op: impl Fn(&F, &F) -> F) -> Self
        where
            F: core::fmt::Debug,
        {
            let dims = self
                .dims
                .iter()
                .zip(other.dims.iter())
                .map(|(a, b)| op(a, b))
                .collect::<Vec<F>>();
            let dims = dims.try_into().unwrap();
            Self { dims }
        }
    }
}

impl<F, const N: usize> Vector<F, N>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.zip_with(other, |&a, &b| a + b)
    }
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.zip_with(other, |&a, &b| a - b)
    }
    #[must_use]
    pub fn dot(&self, other: &Self) -> F {
        self.zip_with(other, |&a, &b| a * b)
            .dims()
            .iter()
            .copied()
            .sum::<F>()
    }
    pub fn mul(&mut self, other: F) {
        self.dims_mut().iter_mut().for_each(|x| *x = *x * other);
    }
    pub fn div(&mut self, other: F) {
        self.mul(F::one() / other);
    }
    #[must_use]
    pub fn mag(&self) -> F {
        let sum = self.dims().iter().map(|x| x.powi(2)).sum::<F>();
        sum.sqrt()
    }
    pub fn normalize(&mut self) {
        self.div(self.mag());
    }
    pub fn set_mag(&mut self, mag: F) {
        self.normalize();
        self.mul(mag);
    }
    pub fn limit(&mut self, min: F, max: F) {
        let mag = self.mag().clamp(min, max);
        self.set_mag(mag);
    }
    #[must_use]
    pub fn dist(&self, other: &Self) -> F {
        self.sub(other).mag()
    }
    #[must_use]
    pub fn lerp(&self, other: &Self, t: NormalizedF64) -> Self {
        self.zip_with(other, |&a, &b| lerp(&(a..=b), t.into()))
    }
    #[must_use]
    pub fn heading_angle(&self, adjacent_axis: usize, opposite_axis: usize) -> F {
        let adj = self.dims()[adjacent_axis];
        let opp = self.dims()[opposite_axis];
        F::atan2(opp, adj)
    }
    pub fn rotate(&mut self, adjacent_axis: usize, opposite_axis: usize, angle: F) {
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
    pub fn angle_between(&self, other: &Self) -> F {
        let dot = self.dot(other);
        let mul_mag = self.mag() * other.mag();
        F::acos(dot / mul_mag)
    }
    #[must_use]
    pub fn normal_point(&self, start: &Self, end: &Self) -> Self {
        let a = self.sub(start);
        let mut b = end.sub(start);
        let angle = a.angle_between(&b);
        let d = a.mag() * F::cos(angle);
        b.set_mag(d);
        b
    }
}
impl<F> Vector<F, 2>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    #[must_use]
    pub fn heading_angle_2d(&self) -> F {
        self.heading_angle(0, 1)
    }
    pub fn rotate_2d(&mut self, angle: F) {
        self.rotate(0, 1, angle)
    }
}
impl<F> Vector<F, 3>
where
    F: Float + core::fmt::Debug + core::iter::Sum,
{
    /// The cross product is only defined in 3D space and takes two non-parallel vectors as input and produces a third vector that is orthogonal to both the input vectors.
    /// - ref: <https://learnopengl.com/Getting-started/Transformations>
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        let dims = [
            self.dims()[1] * other.dims()[2] - self.dims()[2] * other.dims()[1],
            self.dims()[2] * other.dims()[0] - self.dims()[0] * other.dims()[2],
            self.dims()[0] * other.dims()[1] - self.dims()[1] * other.dims()[0],
        ];
        Self::new(dims)
    }
}
impl<F, const N: usize> From<Vector<F, N>> for matrix::ArrayMatrixBuf<F, N>
where
    F: Float,
{
    fn from(value: Vector<F, N>) -> Self {
        let size = matrix::Size {
            rows: NonZeroUsize::new(N).unwrap(),
            cols: NonZeroUsize::new(1).unwrap(),
        };
        matrix::ArrayMatrixBuf::new(size, *value.dims())
    }
}
impl<F, const N: usize> Vector<F, N>
where
    F: Float,
{
    pub fn try_from_matrix<M>(matrix: M) -> Option<Self>
    where
        M: Container2D<F>,
    {
        let size = matrix::Size {
            rows: NonZeroUsize::new(N).unwrap(),
            cols: NonZeroUsize::new(1).unwrap(),
        };
        if matrix.size() != size {
            return None;
        }
        let mut dims = [F::zero(); N];
        for (i, row) in dims.iter_mut().enumerate() {
            let index = matrix::Index { row: i, col: 0 };
            let value = matrix.get(index);
            *row = value;
        }
        Some(Self::new(dims))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use primitive::float::FloatExt;

    use super::*;

    #[test]
    fn test_add() {
        let a = Vector::new([0., 1.]);
        let b = Vector::new([1., 2.]);
        let c = a.add(&b);
        assert_eq!(*c.dims(), [1., 3.]);
    }

    #[test]
    fn test_heading() {
        let v = Vector::new([1., 1.]);
        assert!(v.heading_angle_2d().closes_to(PI / 4.));
        let v = Vector::new([-1., 1.]);
        assert!(v.heading_angle_2d().closes_to(PI / 2. + PI / 4.));
        let v = Vector::new([-1., -1.]);
        assert!(v.heading_angle_2d().closes_to(PI + PI / 4. - 2. * PI));
        let v = Vector::new([1., -1.]);
        assert!(v.heading_angle_2d().closes_to(-PI / 4.));
    }

    #[test]
    fn test_rotation() {
        let mut v = Vector::new([1., 0.5]);
        v.rotate_2d(PI / 2.);
        assert!(v.dims()[0].closes_to(-0.5));
        assert!(v.dims()[1].closes_to(1.));
    }
}
