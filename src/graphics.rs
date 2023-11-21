/// Ref: <https://en.wikipedia.org/wiki/Linear_interpolation>
pub fn lerp(v: &std::ops::RangeInclusive<f64>, t: f64) -> f64 {
    assert!((0.0..=1.0).contains(&t));
    (1.0 - t) * v.start() + t * v.end()
}
