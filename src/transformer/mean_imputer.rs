use std::convert::Infallible;

use crate::statistics::MeanExt;

use super::Transformer;

#[derive(Debug, Clone, Copy)]
pub struct MeanImputer {
    mean: f64,
}

impl Transformer for MeanImputer {
    type Err = Infallible;
    type Value = f64;

    fn transform(&self, x: Self::Value) -> Self::Value {
        if x.is_nan() {
            return self.mean;
        }
        x
    }

    fn fit(examples: impl Iterator<Item = Self::Value> + Clone) -> Result<Self, Self::Err>
    where
        Self: Sized,
    {
        let mean = examples.mean();
        Ok(Self { mean })
    }
}
