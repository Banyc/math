use std::convert::Infallible;

use getset::CopyGetters;

use super::{Estimate, Transform};

#[derive(Debug, Clone, Copy)]
pub struct ProportionScalingEstimator;
impl Estimate for ProportionScalingEstimator {
    type Value = f64;
    type Err = Infallible;
    type Output = ProportionScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let sum = examples.clone().sum();
        Ok(ProportionScaler { sum })
    }
}

#[derive(Debug, Clone, Copy, CopyGetters)]
pub struct ProportionScaler {
    #[getset(get_copy = "pub")]
    sum: f64,
}
impl Transform for ProportionScaler {
    type Value = f64;
    type Err = Infallible;

    fn transform(&self, x: f64) -> f64 {
        x / self.sum
    }
}
