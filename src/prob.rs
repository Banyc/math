#[derive(Debug, Clone, Copy)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        Some(Self(p))
    }

    pub fn certainty() -> Self {
        Self(1.0)
    }

    pub fn impossibility() -> Self {
        Self(0.0)
    }

    pub fn complementary(&self) -> Self {
        Self(1.0 - self.0)
    }

    pub fn get(&self) -> f64 {
        self.0
    }
}

impl PartialEq for Probability {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Probability {}

impl Ord for Probability {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("`Probability` is impossible to be NaN")
    }
}

impl PartialOrd for Probability {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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
}
