use core::ops::RangeInclusive;

use strict_num::{FiniteF64, PositiveF64};
use thiserror::Error;

use super::{Estimate, Transform};

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScalingEstimator {
    range: RangeInclusive<FiniteF64>,
}
impl MinMaxScalingEstimator {
    pub fn new(
        range: RangeInclusive<FiniteF64>,
    ) -> Result<Self, InfiniteOrNegativeGivenRangeLengthError> {
        if PositiveF64::new(range.end().get() - range.start().get()).is_none() {
            return Err(InfiniteOrNegativeGivenRangeLengthError);
        }
        Ok(Self { range })
    }
}
impl Default for MinMaxScalingEstimator {
    fn default() -> Self {
        Self {
            range: FiniteF64::new(0.).unwrap()..=FiniteF64::new(1.).unwrap(),
        }
    }
}
impl Estimate<f64> for MinMaxScalingEstimator {
    type Err = MinMaxScalingEstimateError2;
    type Output = MinMaxScaler;

    fn fit(&self, examples: impl Iterator<Item = f64> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let examples = examples
            .map(FiniteF64::new)
            .collect::<Option<Vec<FiniteF64>>>();
        let examples = examples.ok_or(MinMaxScalingEstimateError2::InvalidInput)?;
        self.fit(examples.iter().copied())
            .map_err(MinMaxScalingEstimateError2::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum MinMaxScalingEstimateError2 {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(MinMaxScalingEstimateError),
}
impl Estimate<FiniteF64> for MinMaxScalingEstimator {
    type Err = MinMaxScalingEstimateError;
    type Output = MinMaxScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = FiniteF64> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for x in examples {
            if x.get() < min {
                min = x.get();
            }
            if x.get() > max {
                max = x.get();
            }
        }

        if max < min {
            return Err(Self::Err::EmptySequenceError);
        };
        if FiniteF64::new(max - min).is_none() {
            return Err(Self::Err::InfiniteMinMaxRange);
        }

        Ok(MinMaxScaler {
            min: FiniteF64::new(min).unwrap(),
            max: FiniteF64::new(max).unwrap(),
            range: self.range.clone(),
        })
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum MinMaxScalingEstimateError {
    #[error("Empty sequence")]
    EmptySequenceError,
    #[error("Infinite min max range")]
    InfiniteMinMaxRange,
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Length of given range is either infinite or negative")]
pub struct InfiniteOrNegativeGivenRangeLengthError;

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScaler {
    min: FiniteF64,
    max: FiniteF64,
    range: core::ops::RangeInclusive<FiniteF64>,
}
impl Transform<f64> for MinMaxScaler {
    type Err = MinMaxScalerError;

    fn transform(&self, x: f64) -> Result<f64, Self::Err> {
        let x = FiniteF64::new(x).ok_or(MinMaxScalerError::InvalidInput)?;
        self.transform(x)
            .map(|x| x.get())
            .map_err(MinMaxScalerError::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum MinMaxScalerError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(OutOfRange),
}
impl Transform<FiniteF64> for MinMaxScaler {
    type Err = OutOfRange;

    fn transform(&self, x: FiniteF64) -> Result<FiniteF64, Self::Err> {
        if !(self.min..=self.max).contains(&x) {
            return Err(OutOfRange);
        }
        let x_std = (x.get() - self.min.get()) / (self.max.get() - self.min.get());
        let scaled =
            x_std * (self.range.end().get() - self.range.start().get()) + self.range.start().get();
        Ok(FiniteF64::new(scaled).unwrap())
    }
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Input is out of min max range")]
pub struct OutOfRange;

#[cfg(test)]
mod tests {
    use crate::transformer::{EstimateExt, TransformExt};

    use super::*;

    #[test]
    fn test() {
        let examples = [-1.0, -0.5, 0.0, 1.0];
        let examples = examples.into_iter().map(|x| FiniteF64::new(x).unwrap());
        let scaler = examples
            .clone()
            .fit(&MinMaxScalingEstimator::default())
            .unwrap();
        assert_eq!(scaler.max, 1.0);
        let transformed = examples.transform_by(scaler);
        let x = transformed.collect::<Result<Vec<FiniteF64>, _>>().unwrap();
        assert_eq!(x, [0.0, 0.25, 0.5, 1.0]);
    }

    #[test]
    fn test_edge() {
        assert!(MinMaxScalingEstimator::new(
            FiniteF64::new(1.0).unwrap()..=FiniteF64::new(0.0).unwrap()
        )
        .is_err());

        let examples = [f64::MIN, f64::MAX];
        let examples = examples.into_iter().map(|x| FiniteF64::new(x).unwrap());
        assert!(examples
            .fit_transform(&MinMaxScalingEstimator::default())
            .is_err());
    }
}
