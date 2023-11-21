pub fn sigmoid(x: f64) -> f64 {
    // Prevent floating point underflow
    let x = x.clamp(-250.0, 250.0);
    1.0 / (1.0 + f64::exp(-x))
}
