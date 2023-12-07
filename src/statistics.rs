use std::num::NonZeroUsize;

pub trait MeanExt: Iterator {
    fn mean(self) -> f64;
}
impl<T> MeanExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn mean(self) -> f64 {
        let n: usize = self.clone().count();
        // Sum of fractions is used to avoid infinite value
        self.map(|x| x / n as f64).sum()
    }
}

pub trait StandardDeviationExt: Iterator {
    fn standard_deviation(self) -> f64;
}
impl<T> StandardDeviationExt for T
where
    T: Iterator<Item = f64> + Clone,
{
    fn standard_deviation(self) -> f64 {
        let mean = self.clone().mean();
        let n: usize = self.clone().count();
        let sum_squared_error: f64 = self.map(|x| (x - mean).powi(2)).sum();
        let variance = sum_squared_error / n as f64;
        variance.sqrt()
    }
}

pub trait DistanceExt: Iterator<Item = (f64, f64)> + Sized {
    fn distance(self, p: NonZeroUsize) -> f64 {
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
