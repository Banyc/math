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

    fn fit(examples: impl Iterator<Item = Self::Value> + Clone) -> Result<Self, Self::Err> {
        let mean = examples.clone().filter(|x| !x.is_nan()).mean();
        Ok(Self { mean })
    }
}

#[cfg(test)]
mod tests {
    use crate::transformer::TransformExt;

    use super::*;

    #[test]
    fn test_missing_numbers() {
        let examples = [2.0, f64::NAN, 5.0];
        let imp_mean = examples.into_iter().fit::<MeanImputer>().unwrap();
        let examples = [2.0, f64::NAN, f64::NAN];
        let transformed = examples.into_iter().transform_by(imp_mean);
        let x = transformed.collect::<Vec<_>>();
        assert_eq!(x, [2.0, 3.5, 3.5]);
    }
}
