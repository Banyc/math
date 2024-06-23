use strict_num::{FiniteF64, NormalizedF64};

use crate::{graphics::lerp, prob::Probability};

#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize> {
    dims: [FiniteF64; N],
}
impl<const N: usize> Vector<N> {
    pub fn new(dims: [FiniteF64; N]) -> Self {
        Self { dims }
    }

    pub fn add(&self, other: &Self) -> Self {
        let dims = self
            .dims
            .iter()
            .copied()
            .zip(other.dims.iter().copied())
            .map(|(a, b)| a.get() + b.get())
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let dims = self
            .dims
            .iter()
            .copied()
            .zip(other.dims.iter().copied())
            .map(|(a, b)| a.get() - b.get())
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
    pub fn dot(&self, other: &Self) -> f64 {
        self.dims
            .iter()
            .copied()
            .zip(other.dims.iter().copied())
            .map(|(a, b)| a.get() * b.get())
            .sum::<f64>()
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
    pub fn dist(&self, other: &Self) -> f64 {
        self.sub(other).mag()
    }
    pub fn lerp(&self, other: &Self, t: NormalizedF64) -> Self {
        let dims = self
            .dims
            .iter()
            .copied()
            .zip(other.dims.iter().copied())
            .map(|(a, b)| lerp(&(a.get()..=b.get()), Probability::new(t.get()).unwrap()))
            .map(|x| FiniteF64::new(x).unwrap())
            .collect::<Vec<FiniteF64>>();
        let dims = dims.try_into().unwrap();
        Self { dims }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Vector::new([0., 1.].map(|x| FiniteF64::new(x).unwrap()));
        let b = Vector::new([1., 2.].map(|x| FiniteF64::new(x).unwrap()));
        let c = a.add(&b);
        assert_eq!(c.dims, [1., 3.].map(|x| FiniteF64::new(x).unwrap()));
    }
}
