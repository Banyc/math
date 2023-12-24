use std::convert::Infallible;

use getset::CopyGetters;
use serde::{Deserialize, Serialize};

use crate::statistics::{EmptySequenceError, MeanExt, StandardDeviationExt};

use super::{Estimate, Transform, Transformed};

pub type Standardized<I> = Transformed<I, StandardScaler>;

#[derive(Debug, Clone, Copy)]
pub struct StandardScalingEstimator;
impl Estimate for StandardScalingEstimator {
    type Value = f64;
    type Err = EmptySequenceError;
    type Output = StandardScaler;

    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let mean = examples.clone().mean()?;
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
    mean: f64,
    #[getset(get_copy = "pub")]
    standard_deviation: f64,
}
impl StandardScaler {
    pub fn new(mean: f64, standard_deviation: f64) -> Self {
        Self {
            mean,
            standard_deviation,
        }
    }
}
impl Transform for StandardScaler {
    type Value = f64;
    type Err = Infallible;

    fn transform(&self, x: f64) -> f64 {
        (x - self.mean()) / self.standard_deviation()
    }
}
