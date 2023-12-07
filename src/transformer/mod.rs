pub mod min_max_scaler;
pub mod standard_scaler;

pub trait Transformer {
    type Err;

    /// Transform a value.
    fn transform(&self, x: f64) -> f64;

    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit(examples: impl Iterator<Item = f64> + Clone) -> Result<Self, Self::Err>
    where
        Self: Sized;
}

pub trait TransformExt: Iterator {
    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit<T: Transformer>(self) -> Result<T, T::Err>;

    /// Map the iterator with a transformer.
    ///
    /// This only transforms the iterator based on the provided transformer,
    /// not on the iterator itself.
    fn transform<T>(self, transformer: T) -> Transformed<Self, T>
    where
        Self: Sized,
        T: Transformer;
}
impl<I: Iterator<Item = f64> + Clone> TransformExt for I {
    fn fit<T: Transformer>(self) -> Result<T, T::Err> {
        T::fit(self)
    }

    fn transform<T>(self, transformer: T) -> Transformed<Self, T>
    where
        Self: Sized,
        T: Transformer,
    {
        Transformed::new(self, transformer)
    }
}

#[derive(Debug, Clone)]
pub struct Transformed<I, T> {
    iter: I,
    transformer: T,
}
impl<I: Iterator<Item = f64>, T: Transformer> Iterator for Transformed<I, T> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| self.transformer.transform(x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I, T> Transformed<I, T> {
    pub fn new(iter: I, transformer: T) -> Self {
        Self { iter, transformer }
    }
}
