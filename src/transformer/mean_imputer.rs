use std::convert::Infallible;

use crate::statistics::{mean::MeanExt, EmptySequenceError};

use super::{Estimate, Transform};

#[derive(Debug, Clone, Copy)]
pub struct MeanImputationEstimator;
impl Estimate<f64> for MeanImputationEstimator {
    type Err = EmptySequenceError;
    type Output = MeanImputer;

    fn fit(&self, examples: impl Iterator<Item = f64> + Clone) -> Result<Self::Output, Self::Err>
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

impl Transform<f64> for MeanImputer {
    type Err = Infallible;

    fn transform(&self, x: f64) -> Result<f64, Self::Err> {
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
        let x = transformed.collect::<Result<Vec<f64>, _>>().unwrap();
        assert_eq!(x, [2.0, 3.5, 3.5]);
    }
}
