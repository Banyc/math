use crate::statistics::mean::MeanExt;

use super::EmptySequenceError;

pub trait VarianceExt: Iterator {
    fn variance(self) -> Result<f64, EmptySequenceError>;
}
impl<T> VarianceExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn variance(self) -> Result<f64, EmptySequenceError> {
        let mean = self.clone().mean()?;
        let n: usize = self.clone().count();
        let variance: f64 = self.map(|x| (x - mean).powi(2) / n as f64).sum();
        Ok(variance)
    }
}
