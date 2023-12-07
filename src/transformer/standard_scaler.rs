use std::convert::Infallible;

use getset::CopyGetters;
use serde::{Deserialize, Serialize};

use crate::statistics::{MeanExt, StandardDeviationExt};

use super::{Transformed, Transformer};

pub type Standardized<I> = Transformed<I, StandardScaler>;

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
impl Transformer for StandardScaler {
    type Value = f64;
    type Err = Infallible;

    fn transform(&self, x: f64) -> f64 {
        (x - self.mean()) / self.standard_deviation()
    }

    fn fit(examples: impl Iterator<Item = f64> + Clone) -> Result<Self, Self::Err> {
        let mean = examples.clone().mean();
        let standard_deviation = examples.standard_deviation();
        Ok(Self {
            mean,
            standard_deviation,
        })
    }
}
