use std::num::NonZeroUsize;

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
        let sum_squared_error: f64 = self.map(|x| (x - mean).powi(2)).sum();
        let variance = sum_squared_error / n as f64;
        Ok(variance.sqrt())
    }
}

pub trait DistanceExt: Iterator<Item = (f64, f64)> + Sized {
    fn distance(self, p: NonZeroUsize) -> f64 {
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
}
