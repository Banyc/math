pub trait FloatExt {
    fn closes_to(self, other: Self) -> bool;
}

const FLOAT_RELATIVE_TOLERANCE: f64 = 1e-9; // for big absolute numbers
const FLOAT_ABSOLUTE_TOLERANCE: f64 = 1e-9; // for near-zero numbers

impl FloatExt for f64 {
    fn closes_to(self, other: Self) -> bool {
        let diff = self - other;
        let tolerance = Self::max(
            FLOAT_RELATIVE_TOLERANCE * Self::max(self.abs(), other.abs()),
            FLOAT_ABSOLUTE_TOLERANCE,
        );
        diff.abs() <= tolerance
    }
}
