use std::num::NonZeroU32;

use strict_num::{FiniteF64, PositiveF64};
use thiserror::Error;

pub trait MeanExt: Iterator {
    fn mean(self) -> Result<f64, EmptySequenceError>;
}
impl<T> MeanExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn mean(self) -> Result<f64, EmptySequenceError> {
        let n: usize = self.clone().count();
        if n == 0 {
            return Err(EmptySequenceError);
        }
        // Sum of fractions is used to avoid infinite value
        Ok(self.map(|x| x / n as f64).sum())
    }
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Empty sequence")]
pub struct EmptySequenceError;

pub trait FiniteMeanExt: Iterator {
    fn mean(self) -> Result<FiniteF64, EmptySequenceError>;
}
impl<T> FiniteMeanExt for T
where
    T: Iterator<Item = FiniteF64> + Clone,
{
    fn mean(self) -> Result<FiniteF64, EmptySequenceError> {
        self.map(|x| x.get())
            .mean()
            .map(|x| FiniteF64::new(x).unwrap())
    }
}

pub trait StandardDeviationExt: Iterator {
    fn standard_deviation(self) -> Result<f64, EmptySequenceError>;
}
impl<T> StandardDeviationExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn standard_deviation(self) -> Result<f64, EmptySequenceError> {
        let mean = self.clone().mean()?;
        let n: usize = self.clone().count();
        let variance: f64 = self.map(|x| (x - mean).powi(2) / n as f64).sum();
        Ok(variance.sqrt())
    }
}

pub trait FiniteStandardDeviationExt: Iterator {
    fn standard_deviation(self) -> Result<PositiveF64, EmptySequenceError>;
}
impl<T> FiniteStandardDeviationExt for T
where
    T: Iterator<Item = FiniteF64> + Clone,
{
    fn standard_deviation(self) -> Result<PositiveF64, EmptySequenceError> {
        self.map(|x| x.get())
            .standard_deviation()
            .map(|x| PositiveF64::new(x).unwrap())
    }
}

pub trait DistanceExt: Iterator<Item = (f64, f64)> + Sized {
    /// # Panic
    ///
    /// If `p` cannot be converted into `i32`.
    fn distance(self, p: NonZeroU32) -> f64 {
        let p_i32 = i32::try_from(p.get()).unwrap();
        let sum = self.map(|(a, b)| (a - b).abs().powi(p_i32)).sum::<f64>();
        match p.get() {
            0 => unreachable!(),
            1 => sum,
            2 => sum.sqrt(),
            _ => {
                let inverse = 1.0 / p.get() as f64;
                sum.powf(inverse)
            }
        }
    }
}
impl<T: Iterator<Item = (f64, f64)>> DistanceExt for T {}

pub trait FiniteDistanceExt: Iterator<Item = (FiniteF64, FiniteF64)> + Sized {
    /// # Panic
    ///
    /// If `p` cannot be converted into `i32`.
    fn distance(self, p: NonZeroU32) -> Result<FiniteF64, InfiniteDistanceError> {
        let d = self.map(|(a, b)| (a.get(), b.get())).distance(p);
        FiniteF64::new(d).ok_or(InfiniteDistanceError)
    }
}
impl<T: Iterator<Item = (FiniteF64, FiniteF64)>> FiniteDistanceExt for T {}
#[derive(Debug, Error, Clone, Copy)]
#[error("Infinite distance")]
pub struct InfiniteDistanceError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let polarized = [f64::NEG_INFINITY, f64::INFINITY];
        let mean = polarized.iter().copied().mean().unwrap();
        assert!(mean.is_nan());

        let empty: [f64; 0] = [];
        assert!(empty.iter().copied().mean().is_err());
    }

    #[test]
    fn test_distance() {
        let polarized = [(
            FiniteF64::new(f64::MIN).unwrap(),
            FiniteF64::new(f64::MAX).unwrap(),
        )];
        let distance = polarized
            .iter()
            .copied()
            .distance(NonZeroU32::new(2).unwrap());
        assert!(distance.is_err());
    }
}
