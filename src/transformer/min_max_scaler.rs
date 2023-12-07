use thiserror::Error;

use super::Transformer;

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

    fn fit(examples: impl Iterator<Item = f64> + Clone) -> Result<Self, Self::Err>
    where
        Self: Sized,
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for example in examples {
            if example.is_nan() {
                return Err(MinMaxScalerError::MissingNumber);
            }
            if example < min {
                min = example;
            }
            if example > max {
                max = example;
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
