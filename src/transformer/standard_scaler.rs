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
impl Estimate for StandardScalingEstimator {
    type Value = FiniteF64;
    type Err = StandardDeviationError;
    type Output = StandardScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
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
impl Transform for StandardScaler {
    type Value = FiniteF64;
    type Err = InfiniteStandardizedNum;

    fn transform(&self, x: Self::Value) -> Result<Self::Value, Self::Err> {
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
