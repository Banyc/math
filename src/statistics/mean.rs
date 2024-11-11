use crate::R;

use super::EmptySequenceError;

pub trait MeanExt: Iterator {
    fn mean(self) -> Result<f64, EmptySequenceError>;
}
impl<T> MeanExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn mean(self) -> Result<f64, EmptySequenceError> {
        let n: usize = self.clone().count();
        if n == 0 {
            return Err(EmptySequenceError);
        }
        // Sum of fractions is used to avoid infinite value
        Ok(self.map(|x| x / n as f64).sum())
    }
}

pub trait FiniteMeanExt: Iterator {
    fn mean(self) -> Result<R<f64>, EmptySequenceError>;
}
impl<T> FiniteMeanExt for T
where
    T: Iterator<Item = R<f64>> + Clone,
{
    fn mean(self) -> Result<R<f64>, EmptySequenceError> {
        self.map(|x| x.get()).mean().map(|x| R::new(x).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let polarized = [f64::NEG_INFINITY, f64::INFINITY];
        let mean = polarized.iter().copied().mean().unwrap();
        assert!(mean.is_nan());

        let empty: [f64; 0] = [];
        assert!(empty.iter().copied().mean().is_err());
    }
}
