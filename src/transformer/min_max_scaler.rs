use thiserror::Error;

use super::Transformer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinMaxScaler {
    min: f64,
    max: f64,
}
impl Transformer for MinMaxScaler {
    type Value = f64;
    type Err = MinMaxScalerError;

    fn transform(&self, x: f64) -> f64 {
        (x - self.min) / (self.max - self.min)
    }

    fn fit(examples: impl Iterator<Item = f64> + Clone) -> Result<Self, Self::Err> {
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

        Ok(Self { min, max })
    }
}

#[derive(Debug, Clone, Error)]
pub enum MinMaxScalerError {
    #[error("A number is NAN in the iterator")]
    MissingNumber,
    #[error("No number in the iterator")]
    NoNumber,
}
