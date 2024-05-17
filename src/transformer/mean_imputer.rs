use std::convert::Infallible;

use crate::statistics::{mean::MeanExt, EmptySequenceError};

use super::{Estimate, Transform};

#[derive(Debug, Clone, Copy)]
pub struct MeanImputationEstimator;
impl Estimate for MeanImputationEstimator {
    type Err = EmptySequenceError;
    type Value = f64;
    type Output = MeanImputer;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let mean = examples.clone().filter(|x| !x.is_nan()).mean()?;
        Ok(MeanImputer { mean })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MeanImputer {
    mean: f64,
}

impl Transform for MeanImputer {
    type Err = Infallible;
    type Value = f64;

    fn transform(&self, x: Self::Value) -> Result<Self::Value, Self::Err> {
        if x.is_nan() {
            return Ok(self.mean);
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::transformer::{EstimateExt, TransformExt};

    use super::*;

    #[test]
    fn test_missing_numbers() {
        let examples = [2.0, f64::NAN, 5.0];
        let imp_mean = examples.into_iter().fit(&MeanImputationEstimator).unwrap();
        let examples = [2.0, f64::NAN, f64::NAN];
        let transformed = examples.into_iter().transform_by(imp_mean);
        let x = transformed.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(x, [2.0, 3.5, 3.5]);
    }
}
