use core::num::NonZeroU32;

use strict_num::FiniteF64;
use thiserror::Error;

pub trait DistanceExt: Iterator<Item = (f64, f64)> + Sized {
    /// # Panic
    ///
    /// If `p` cannot be converted into `i32`.
    fn distance(self, p: NonZeroU32) -> f64 {
        let p_i32 = i32::try_from(p.get()).unwrap();
        let sum = self.map(|(a, b)| (a - b).abs().powi(p_i32)).sum::<f64>();
        match p.get() {
            0 => unreachable!(),
            1 => sum,
            2 => sum.sqrt(),
            _ => {
                let inverse = 1.0 / p.get() as f64;
                sum.powf(inverse)
            }
        }
    }
}
impl<T: Iterator<Item = (f64, f64)>> DistanceExt for T {}

pub trait FiniteDistanceExt: Iterator<Item = (FiniteF64, FiniteF64)> + Sized {
    /// # Panic
    ///
    /// If `p` cannot be converted into `i32`.
    fn distance(self, p: NonZeroU32) -> Result<FiniteF64, InfiniteDistanceError> {
        let d = self.map(|(a, b)| (a.get(), b.get())).distance(p);
        FiniteF64::new(d).ok_or(InfiniteDistanceError)
    }
}
impl<T: Iterator<Item = (FiniteF64, FiniteF64)>> FiniteDistanceExt for T {}
#[derive(Debug, Error, Clone, Copy)]
#[error("Infinite distance")]
pub struct InfiniteDistanceError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        let polarized = [(
            FiniteF64::new(f64::MIN).unwrap(),
            FiniteF64::new(f64::MAX).unwrap(),
        )];
        let distance = polarized
            .iter()
            .copied()
            .distance(NonZeroU32::new(2).unwrap());
        assert!(distance.is_err());
    }
}
