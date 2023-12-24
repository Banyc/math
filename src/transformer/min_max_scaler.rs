use std::{convert::Infallible, ops::RangeInclusive};

use strict_num::FiniteF64;

use crate::statistics::EmptySequenceError;

use super::{Estimate, Transform};

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScalingEstimator {
    range: RangeInclusive<FiniteF64>,
}
impl MinMaxScalingEstimator {
    pub fn new(range: RangeInclusive<FiniteF64>) -> Self {
        Self { range }
    }
}
impl Default for MinMaxScalingEstimator {
    fn default() -> Self {
        Self {
            range: FiniteF64::new(0.).unwrap()..=FiniteF64::new(1.).unwrap(),
        }
    }
}
impl Estimate for MinMaxScalingEstimator {
    type Err = EmptySequenceError;
    type Value = FiniteF64;
    type Output = MinMaxScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
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
            return Err(EmptySequenceError);
        };

        Ok(MinMaxScaler {
            min: FiniteF64::new(min).unwrap(),
            max: FiniteF64::new(max).unwrap(),
            range: self.range.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MinMaxScaler {
    min: FiniteF64,
    max: FiniteF64,
    range: std::ops::RangeInclusive<FiniteF64>,
}
impl Transform for MinMaxScaler {
    type Value = FiniteF64;
    type Err = Infallible;

    fn transform(&self, x: Self::Value) -> Result<Self::Value, Self::Err> {
        let x_std = (x.get() - self.min.get()) / (self.max.get() - self.min.get());
        let scaled =
            x_std * (self.range.end().get() - self.range.start().get()) + self.range.start().get();
        Ok(FiniteF64::new(scaled).unwrap())
    }
}

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
        let x = transformed.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(x, [0.0, 0.25, 0.5, 1.0]);
    }
}
