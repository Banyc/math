use primitive::ops::float::FloatExt;
use strict_num::NormalizedF64;

#[derive(Debug, Clone, Copy)]
pub struct Probability(NormalizedF64);
impl Probability {
    /// # Option
    ///
    /// Return [`None`] if `p` is not in `[0, 1]`
    #[must_use]
    pub fn new(p: f64) -> Option<Self> {
        let p = NormalizedF64::new(p)?;
        Some(Self(p))
    }
    #[must_use]
    pub fn certainty() -> Self {
        Self(NormalizedF64::new(1.0).unwrap())
    }
    #[must_use]
    pub fn impossibility() -> Self {
        Self(NormalizedF64::new(0.0).unwrap())
    }
    #[must_use]
    pub fn complementary(&self) -> Self {
        Self(NormalizedF64::new(1.0 - self.0.get()).unwrap())
    }
    #[must_use]
    pub fn get(&self) -> f64 {
        self.0.get()
    }
}
impl PartialEq for Probability {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Probability {}
impl Ord for Probability {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("`Probability` is impossible to be NaN")
    }
}
impl PartialOrd for Probability {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl core::ops::Mul for Probability {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.get() * rhs.get()).unwrap()
    }
}
impl From<Probability> for NormalizedF64 {
    fn from(value: Probability) -> Self {
        NormalizedF64::new(value.get()).unwrap()
    }
}
impl From<NormalizedF64> for Probability {
    fn from(value: NormalizedF64) -> Self {
        Self::new(value.get()).unwrap()
    }
}

pub trait WeightedSumExt: Iterator<Item = (Probability, f64)> {
    /// # Option
    ///
    /// Return [`None`] if the sum of the weights is not equal to 1
    fn weighted_sum(self) -> Option<f64>
    where
        Self: Sized,
    {
        let mut prob_sum = 0.0;

        let weighted_sum = self
            .map(|(p, x)| {
                prob_sum += p.get();
                p.get() * x
            })
            .sum();

        // Make sure the sum of the weights is 1
        if !prob_sum.closes_to(Probability::certainty().get()) {
            return None;
        };

        Some(weighted_sum)
    }
}
impl<I: Iterator<Item = (Probability, f64)>> WeightedSumExt for I {}

#[derive(Debug, Clone)]
pub struct Fraction<I> {
    iter: I,
    sum: f64,
}
impl<I: Iterator<Item = f64>> Iterator for Fraction<I> {
    type Item = Option<Probability>;
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.iter.next()?;
        Some(Probability::new(x / self.sum))
    }
}
pub trait FractionExt: Iterator + Sized {
    fn fraction(self) -> Fraction<Self>;
}
impl<I: Iterator<Item = f64> + Clone> FractionExt for I {
    fn fraction(self) -> Fraction<Self> {
        let sum = self.clone().sum();
        Fraction { iter: self, sum }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability() {
        assert!(Probability::new(f64::NAN).is_none());
        assert!(Probability::new(f64::INFINITY).is_none());
        assert!(Probability::new(-1.0).is_none());
        assert!(Probability::new(1.1).is_none());
        assert_eq!(Probability::new(1.0).unwrap().get(), 1.0);
        assert_eq!(Probability::new(0.0).unwrap().get(), 0.0);
        assert_eq!(Probability::new(-0.0).unwrap().get(), 0.0);
        assert_eq!(
            Probability::certainty().complementary(),
            Probability::impossibility()
        );
    }

    #[test]
    fn test_weighted_sum() {
        let data = [(0.2, 25.0), (0.15, 20.0), (0.4, 15.0), (0.25, 30.0)];
        let weighted_sum = data
            .into_iter()
            .map(|(p, x)| (Probability::new(p).unwrap(), x))
            .weighted_sum();
        assert!(weighted_sum.unwrap().closes_to(21.5));
    }

    #[test]
    fn test_probability_mul() {
        let a = Probability::new(f64::MIN_POSITIVE).unwrap();
        let b = Probability::new(f64::MIN_POSITIVE).unwrap();
        let c = a * b;
        assert_eq!(c, Probability::impossibility());
    }
}
