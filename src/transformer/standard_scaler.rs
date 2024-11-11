use getset::CopyGetters;
use crate::{NonNegR, R};
use thiserror::Error;

use crate::statistics::{
    mean::FiniteMeanExt,
    standard_deviation::{FiniteStandardDeviationExt, StandardDeviationError},
};

use super::{Estimate, Transform, Transformed};

pub type Standardized<I> = Transformed<I, StandardScaler>;

#[derive(Debug, Clone, Copy)]
pub struct StandardScalingEstimator;
impl Estimate<f64> for StandardScalingEstimator {
    type Err = StandardScalingEstimatorError;
    type Output = StandardScaler;

    fn fit(&self, examples: impl Iterator<Item = f64> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let examples = examples.map(R::new).collect::<Option<Vec<R<f64>>>>();
        let examples = examples.ok_or(StandardScalingEstimatorError::InvalidInput)?;
        self.fit(examples.iter().copied())
            .map_err(StandardScalingEstimatorError::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum StandardScalingEstimatorError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(StandardDeviationError),
}
impl Estimate<R<f64>> for StandardScalingEstimator {
    type Err = StandardDeviationError;
    type Output = StandardScaler;

    fn fit(&self, examples: impl Iterator<Item = R<f64>> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let Ok(mean) = examples.clone().mean() else {
            return Err(Self::Err::EmptySequence);
        };
        let standard_deviation = examples.standard_deviation()?;
        Ok(StandardScaler {
            mean,
            standard_deviation,
        })
    }
}

#[derive(Debug, Clone, Copy, CopyGetters)]
/// Borrowed from `sklearn.preprocessing.StandardScaler` but only for one feature.
pub struct StandardScaler {
    #[getset(get_copy = "pub")]
    mean: R<f64>,
    #[getset(get_copy = "pub")]
    standard_deviation: NonNegR<f64>,
}
impl StandardScaler {
    pub fn new(mean: R<f64>, standard_deviation: NonNegR<f64>) -> Self {
        Self {
            mean,
            standard_deviation,
        }
    }
}
impl Transform<f64> for StandardScaler {
    type Err = StandardScalerError;

    fn transform(&self, x: f64) -> Result<f64, Self::Err> {
        let x = R::new(x).ok_or(StandardScalerError::InvalidInput)?;
        self.transform(x)
            .map(|x| x.get())
            .map_err(StandardScalerError::Inner)
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum StandardScalerError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("{0}")]
    Inner(InfiniteStandardizedNum),
}
impl Transform<R<f64>> for StandardScaler {
    type Err = InfiniteStandardizedNum;

    fn transform(&self, x: R<f64>) -> Result<R<f64>, Self::Err> {
        let scaled = x.get() / self.standard_deviation().get()
            - self.mean().get() / self.standard_deviation().get();
        let Some(scaled) = R::new(scaled) else {
            return Err(InfiniteStandardizedNum);
        };
        Ok(scaled)
    }
}
#[derive(Debug, Error, Clone, Copy)]
#[error("Standardized number is infinite")]
pub struct InfiniteStandardizedNum;

#[cfg(test)]
mod tests {
    use crate::NonNegR;

    use super::*;

    #[test]
    fn test_standard_scaler() {
        let sc = StandardScaler::new(
            R::new(f64::MIN).unwrap(),
            NonNegR::new(f64::MIN_POSITIVE).unwrap(),
        );
        assert!(sc.transform(R::new(f64::MAX).unwrap()).is_err());
    }
}
