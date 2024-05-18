use getset::CopyGetters;
use strict_num::PositiveF64;
use thiserror::Error;

use super::{Estimate, Transform};

#[derive(Debug, Clone, Copy)]
pub struct ProportionScalingEstimator;
impl Estimate<f64> for ProportionScalingEstimator {
    type Err = ProportionScalingEstimatorError;
    type Output = ProportionScaler;

    fn fit(&self, examples: impl Iterator<Item = f64> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let examples = examples
            .map(PositiveF64::new)
            .collect::<Option<Vec<PositiveF64>>>();
        let examples = examples.ok_or(ProportionScalingEstimatorError::InvalidInput)?;
        self.fit(examples.iter().copied())
            .map_err(ProportionScalingEstimatorError::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum ProportionScalingEstimatorError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(InfiniteSum),
}
impl Estimate<PositiveF64> for ProportionScalingEstimator {
    type Err = InfiniteSum;
    type Output = ProportionScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = PositiveF64> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let sum = examples.map(|x| x.get()).sum();
        let Some(sum) = PositiveF64::new(sum) else {
            return Err(InfiniteSum);
        };
        Ok(ProportionScaler { sum })
    }
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Infinite sum")]
pub struct InfiniteSum;

#[derive(Debug, Clone, Copy, CopyGetters)]
pub struct ProportionScaler {
    #[getset(get_copy = "pub")]
    sum: PositiveF64,
}
impl Transform<f64> for ProportionScaler {
    type Err = ProportionScalerError;

    fn transform(&self, x: f64) -> Result<f64, Self::Err> {
        let x = PositiveF64::new(x).ok_or(ProportionScalerError::InvalidInput)?;
        self.transform(x)
            .map(|x| x.get())
            .map_err(ProportionScalerError::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum ProportionScalerError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(GreaterThanSum),
}
impl Transform<PositiveF64> for ProportionScaler {
    type Err = GreaterThanSum;

    fn transform(&self, x: PositiveF64) -> Result<PositiveF64, Self::Err> {
        if self.sum.get() < x.get() {
            return Err(GreaterThanSum);
        }
        let scaled = x.get() / self.sum.get();
        Ok(PositiveF64::new(scaled).unwrap())
    }
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Input is greater than sum")]
pub struct GreaterThanSum;
