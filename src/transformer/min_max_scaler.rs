use std::ops::RangeInclusive;

use thiserror::Error;

use super::{Estimate, Transform};

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScalingEstimator {
    range: RangeInclusive<f64>,
}
impl MinMaxScalingEstimator {
    pub fn new(range: RangeInclusive<f64>) -> Self {
        Self { range }
    }
}
impl Default for MinMaxScalingEstimator {
    fn default() -> Self {
        Self { range: 0. ..=1. }
    }
}
impl Estimate for MinMaxScalingEstimator {
    type Err = MinMaxScalerError;
    type Value = f64;
    type Output = MinMaxScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for x in examples {
            if x.is_nan() {
                return Err(MinMaxScalerError::MissingNumber);
            }
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }

        if max < min {
            return Err(MinMaxScalerError::NoNumber);
        };

        Ok(MinMaxScaler {
            min,
            max,
            range: self.range.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScaler {
    min: f64,
    max: f64,
    range: std::ops::RangeInclusive<f64>,
}
impl Transform for MinMaxScaler {
    type Value = f64;
    type Err = MinMaxScalerError;

    fn transform(&self, x: f64) -> f64 {
        let x_std = (x - self.min) / (self.max - self.min);
        x_std * (self.range.end() - self.range.start()) + self.range.start()
    }
}

#[derive(Debug, Clone, Error)]
pub enum MinMaxScalerError {
    #[error("A number is NAN in the iterator")]
    MissingNumber,
    #[error("No number in the iterator")]
    NoNumber,
}

#[cfg(test)]
mod tests {
    use crate::transformer::{EstimateExt, TransformExt};

    use super::*;

    #[test]
    fn test() {
        let examples = [-1.0, -0.5, 0.0, 1.0];
        let scaler = examples
            .into_iter()
            .fit(&MinMaxScalingEstimator::default())
            .unwrap();
        assert_eq!(scaler.max, 1.0);
        let transformed = examples.into_iter().transform_by(scaler);
        let x = transformed.collect::<Vec<_>>();
        assert_eq!(x, [0.0, 0.25, 0.5, 1.0]);
    }
}
