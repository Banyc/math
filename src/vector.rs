use strict_num::{FiniteF64, NormalizedF64};

use crate::graphics::lerp;

#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize> {
    dims: [FiniteF64; N],
}
impl<const N: usize> Vector<N> {
    #[must_use]
    pub fn new(dims: [FiniteF64; N]) -> Self {
        Self { dims }
    }
    #[must_use]
    pub fn dims(&self) -> &[FiniteF64; N] {
        &self.dims
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let dims = self
            .pairwise(other)
            .map(|(a, b)| a.get() + b.get())
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let dims = self
            .pairwise(other)
            .map(|(a, b)| a.get() - b.get())
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.pairwise(other)
            .map(|(a, b)| a.get() * b.get())
            .sum::<f64>()
    }
    fn pairwise<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (FiniteF64, FiniteF64)> + 'a {
        self.dims.iter().copied().zip(other.dims.iter().copied())
    }
    pub fn mul(&mut self, other: f64) {
        self.dims
            .iter_mut()
            .map(|x| (x.get() * other, x))
            .for_each(|(res, x)| *x = FiniteF64::new(res).unwrap());
    }
    pub fn div(&mut self, other: f64) {
        self.mul(1. / other);
    }
    #[must_use]
    pub fn mag(&self) -> f64 {
        let sum = self.dims.iter().map(|x| x.get().powi(2)).sum::<f64>();
        sum.sqrt()
    }
    pub fn normalize(&mut self) {
        self.div(self.mag());
    }
    pub fn set_mag(&mut self, mag: f64) {
        self.normalize();
        self.mul(mag);
    }
    pub fn limit(&mut self, min: f64, max: f64) {
        let mag = self.mag().clamp(min, max);
        self.set_mag(mag);
    }
    #[must_use]
    pub fn dist(&self, other: &Self) -> f64 {
        self.sub(other).mag()
    }
    #[must_use]
    pub fn lerp(&self, other: &Self, t: NormalizedF64) -> Self {
        let dims = self
            .pairwise(other)
            .map(|(a, b)| lerp(&(a.get()..=b.get()), t.into()))
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
    #[must_use]
    pub fn heading_angle(&self, adjacent_axis: usize, opposite_axis: usize) -> f64 {
        let adj = self.dims[adjacent_axis];
        let opp = self.dims[opposite_axis];
        f64::atan2(opp.get(), adj.get())
    }
    pub fn rotate(&mut self, adjacent_axis: usize, opposite_axis: usize, angle: f64) {
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.dims[adjacent_axis];
        let y = self.dims[opposite_axis];
        let x_ = cos * x.get() + -sin * y.get();
        let y_ = sin * x.get() + cos * y.get();
        self.dims[adjacent_axis] = FiniteF64::new(x_).unwrap();
        self.dims[opposite_axis] = FiniteF64::new(y_).unwrap();
    }
    #[must_use]
    pub fn angle_between(&self, other: &Self) -> f64 {
        let dot = self.dot(other);
        let mul_mag = self.mag() * other.mag();
        f64::acos(dot / mul_mag)
    }
    #[must_use]
    pub fn normal_point(&self, start: &Self, end: &Self) -> Self {
        let a = self.sub(start);
        let mut b = end.sub(start);
        let angle = a.angle_between(&b);
        let d = a.mag() * f64::cos(angle);
        b.set_mag(d);
        b
    }
}
impl Vector<2> {
    #[must_use]
    pub fn heading_angle_2d(&self) -> f64 {
        self.heading_angle(0, 1)
    }
    pub fn rotate_2d(&mut self, angle: f64) {
        self.rotate(0, 1, angle)
    }
}
impl Vector<3> {
    /// The cross product is only defined in 3D space and takes two non-parallel vectors as input and produces a third vector that is orthogonal to both the input vectors.
    /// - ref: <https://learnopengl.com/Getting-started/Transformations>
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        let dims = [
            FiniteF64::new(
                self.dims[1].get() * other.dims[2].get() - self.dims[2].get() * other.dims[1].get(),
            )
            .unwrap(),
            FiniteF64::new(
                self.dims[2].get() * other.dims[0].get() - self.dims[0].get() * other.dims[2].get(),
            )
            .unwrap(),
            FiniteF64::new(
                self.dims[0].get() * other.dims[1].get() - self.dims[1].get() * other.dims[0].get(),
            )
            .unwrap(),
        ];
        Self { dims }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::float::FloatExt;

    use super::*;

    #[test]
    fn test_add() {
        let a = Vector::new([0., 1.].map(|x| FiniteF64::new(x).unwrap()));
        let b = Vector::new([1., 2.].map(|x| FiniteF64::new(x).unwrap()));
        let c = a.add(&b);
        assert_eq!(c.dims, [1., 3.].map(|x| FiniteF64::new(x).unwrap()));
    }

    #[test]
    fn test_heading() {
        let v = Vector::new([1., 1.].map(|x| FiniteF64::new(x).unwrap()));
        assert!(v.heading_angle_2d().closes_to(PI / 4.));
        let v = Vector::new([-1., 1.].map(|x| FiniteF64::new(x).unwrap()));
        assert!(v.heading_angle_2d().closes_to(PI / 2. + PI / 4.));
        let v = Vector::new([-1., -1.].map(|x| FiniteF64::new(x).unwrap()));
        assert!(v.heading_angle_2d().closes_to(PI + PI / 4. - 2. * PI));
        let v = Vector::new([1., -1.].map(|x| FiniteF64::new(x).unwrap()));
        assert!(v.heading_angle_2d().closes_to(-PI / 4.));
    }

    #[test]
    fn test_rotation() {
        let mut v = Vector::new([1., 0.5].map(|x| FiniteF64::new(x).unwrap()));
        v.rotate_2d(PI / 2.);
        assert!(v.dims[0].get().closes_to(-0.5));
        assert!(v.dims[1].get().closes_to(1.));
    }
}
