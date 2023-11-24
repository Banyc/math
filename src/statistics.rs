use getset::CopyGetters;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Standardized<I> {
    iter: I,
    sc: StandardScaler,
}
impl<I: Iterator<Item = f64>> Iterator for Standardized<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(move |x| (x - self.sc.mean()) / self.sc.standard_deviation())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I> Standardized<I> {
    pub fn new(iter: I, sc: StandardScaler) -> Self {
        Self { iter, sc }
    }
}

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

pub trait MeanExt: Iterator {
    fn mean(self) -> f64;
}
impl<T> MeanExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn mean(self) -> f64 {
        let n: usize = self.clone().count();
        // Sum of fractions is used to avoid infinite value
        self.map(|x| x / n as f64).sum()
    }
}

pub trait StandardDeviationExt: Iterator {
    fn standard_deviation(self) -> f64;
}
impl<T> StandardDeviationExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn standard_deviation(self) -> f64 {
        let mean = self.clone().mean();
        let n: usize = self.clone().count();
        let sum_squared_error: f64 = self.map(|x| (x - mean).powi(2)).sum();
        let variance = sum_squared_error / n as f64;
        variance.sqrt()
    }
}
