use getset::CopyGetters;
use serde::{Deserialize, Serialize};
use strict_num::{FiniteF64, PositiveF64};
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
        let examples = examples
            .map(FiniteF64::new)
            .collect::<Option<Vec<FiniteF64>>>();
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
impl Estimate<FiniteF64> for StandardScalingEstimator {
    type Err = StandardDeviationError;
    type Output = StandardScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = FiniteF64> + Clone,
    ) -> Result<Self::Output, Self::Err>
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, CopyGetters)]
/// Borrowed from `sklearn.preprocessing.StandardScaler` but only for one feature.
pub struct StandardScaler {
    #[getset(get_copy = "pub")]
    mean: FiniteF64,
    #[getset(get_copy = "pub")]
    standard_deviation: PositiveF64,
}
impl StandardScaler {
    pub fn new(mean: FiniteF64, standard_deviation: PositiveF64) -> Self {
        Self {
            mean,
            standard_deviation,
        }
    }
}
impl Transform<f64> for StandardScaler {
    type Err = StandardScalerError;

    fn transform(&self, x: f64) -> Result<f64, Self::Err> {
        let x = FiniteF64::new(x).ok_or(StandardScalerError::InvalidInput)?;
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
impl Transform<FiniteF64> for StandardScaler {
    type Err = InfiniteStandardizedNum;

    fn transform(&self, x: FiniteF64) -> Result<FiniteF64, Self::Err> {
        let scaled = x.get() / self.standard_deviation().get()
            - self.mean().get() / self.standard_deviation().get();
        let Some(scaled) = FiniteF64::new(scaled) else {
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
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let sc = StandardScaler::new(
            FiniteF64::new(f64::MIN).unwrap(),
            PositiveF64::new(f64::MIN_POSITIVE).unwrap(),
        );
        assert!(sc.transform(FiniteF64::new(f64::MAX).unwrap()).is_err());
    }
}
