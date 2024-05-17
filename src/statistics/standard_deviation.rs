use strict_num::{FiniteF64, PositiveF64};
use thiserror::Error;

use crate::statistics::mean::MeanExt;

use super::EmptySequenceError;

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
    fn standard_deviation(self) -> Result<PositiveF64, StandardDeviationError>;
}
impl<T> FiniteStandardDeviationExt for T
where
    T: Iterator<Item = FiniteF64> + Clone,
{
    fn standard_deviation(self) -> Result<PositiveF64, StandardDeviationError> {
        let std_dev = self
            .map(|x| x.get())
            .standard_deviation()
            .map_err(|_| StandardDeviationError::EmptySequence)?;
        let Some(std_dev) = PositiveF64::new(std_dev) else {
            return Err(StandardDeviationError::InfiniteNum);
        };
        Ok(std_dev)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum StandardDeviationError {
    #[error("Empty sequence")]
    EmptySequence,
    #[error("Infinite number")]
    InfiniteNum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_dev() {
        let polarized = std::iter::repeat(f64::MIN)
            .take(1 << 5)
            .chain(std::iter::once(f64::MAX))
            .map(|x| FiniteF64::new(x).unwrap());
        let std_dev = polarized.standard_deviation();
        assert!(std_dev.is_err());
    }
}
