use std::convert::Infallible;

use getset::CopyGetters;
use serde::{Deserialize, Serialize};

use crate::statistics::{MeanExt, StandardDeviationExt};

use super::{Transformed, Transformer};

pub type Standardized<I> = Transformed<I, StandardScaler>;

pub trait StandardizedExt: Iterator {
    /// Standardizes the iterator based on the iterator itself.
    fn standardized(self) -> Standardized<Self>
    where
        Self: Clone;
    /// Fits a standard scaler from the elements of the iterator,
    /// so that you can use this scaler to standardize another iterator.
    fn standard_scaler(self) -> StandardScaler
    where
        Self: Clone;
    /// Standardizes the iterator with a standard scaler.
    ///
    /// This only scales the iterator based on the provided `sc`,
    /// not on the iterator itself.
    fn standardized_with(self, sc: StandardScaler) -> Standardized<Self>
    where
        Self: Sized;
}
impl<T> StandardizedExt for T
where
    T: Iterator<Item = f64>,
{
    fn standardized(self) -> Standardized<Self>
    where
        Self: Clone,
    {
        let sc = self.clone().standard_scaler();
        self.standardized_with(sc)
    }

    fn standard_scaler(self) -> StandardScaler
    where
        Self: Clone,
    {
        StandardScaler::fit(self)
    }

    fn standardized_with(self, sc: StandardScaler) -> Standardized<Self> {
        Standardized::new(self, sc)
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

    /// Fits a standard scaler from the elements of the iterator,
    /// so that you can use this scaler to standardize another iterator.
    pub fn fit(examples: impl Iterator<Item = f64> + Clone) -> Self {
        let mean = examples.clone().mean();
        let standard_deviation = examples.standard_deviation();
        Self {
            mean,
            standard_deviation,
        }
    }
}
impl Transformer for StandardScaler {
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
