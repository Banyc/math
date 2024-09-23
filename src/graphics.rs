use num_traits::Float;
use strict_num::NormalizedF64;

use crate::prob::Probability;

/// Ref: <https://en.wikipedia.org/wiki/Linear_interpolation>
#[must_use]
pub fn lerp<F: Float>(v: &core::ops::RangeInclusive<F>, t: Probability) -> F {
    let t_compl = F::from(t.complementary().get()).unwrap();
    let t = F::from(t.get()).unwrap();
    t_compl * *v.start() + t * *v.end()
}

#[must_use]
pub fn perlin_interpolation(t: Probability) -> Probability {
    let a = 6. * t.get().powi(5);
    let b = -15. * t.get().powi(4);
    let c = 10. * t.get().powi(3);
    let g = a + b + c;
    NormalizedF64::new_clamped(g).into()
}

/// Generate n colors with equally spaced hues.
///
/// Ref: <https://github.com/scikit-learn/scikit-learn/blob/94f0d6aa7b2d3bdc3d60507daca9b83c7e8b7633/sklearn/tree/_export.py#L27>
pub fn brew_colors(n: usize) -> Vec<(u8, u8, u8)> {
    // Saturation
    const S: f64 = 0.75;
    // Value
    const V: f64 = 0.9;
    // Chroma
    const C: f64 = S * V;
    // Value shift
    const M: f64 = V - C;

    let mut colors = Vec::with_capacity(n);

    const RANGE: core::ops::Range<f64> = 25.0..385.0;
    const DIFF: f64 = RANGE.end - RANGE.start;
    let step = DIFF / n as f64;

    for h in (0..n).map(|i| i as f64 * step) {
        // Calculate some intermediate values
        let h_bar = h / 60.0;
        let x = C * (1.0 - ((h_bar % 2.0) - 1.0).abs());
        // Initialize RGB with same hue & chroma as our color
        let rgb = [
            (C, x, 0.),
            (x, C, 0.),
            (0., C, x),
            (0., x, C),
            (x, 0., C),
            (C, 0., x),
            (C, x, 0.),
        ];
        let (r, g, b) = rgb[h_bar as usize];
        // Shift the initial RGB values to match value and store
        let shift = |primary: f64| -> u8 {
            let primary = u8::MAX as f64 * (primary + M);
            primary as u8
        };
        colors.push((shift(r), shift(g), shift(b)));
    }

    colors
}
