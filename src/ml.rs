use crate::prob::Probability;

/// # Panic
///
/// If `x` is [`std::f64::NAN`]
pub fn sigmoid(x: f64) -> Probability {
    assert!(!x.is_nan());
    // Prevent floating point underflow
    let x = x.clamp(-250.0, 250.0);
    let s = 1.0 / (1.0 + f64::exp(-x));
    Probability::new(s).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::float_ext::FloatExt;

    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = 300.0;
        assert!(sigmoid(x).get().closes_to(1.0));
        let x = -300.0;
        assert!(sigmoid(x).get().closes_to(0.0));
        let x = 0.0;
        assert!(sigmoid(x).get().closes_to(0.5));
        let x = f64::INFINITY;
        assert!(sigmoid(x).get().closes_to(1.0));
        let x = -f64::INFINITY;
        assert!(sigmoid(x).get().closes_to(0.0));
    }
}
