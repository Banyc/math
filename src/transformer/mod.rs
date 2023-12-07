pub mod mean_imputer;
pub mod min_max_scaler;
pub mod standard_scaler;

pub trait Transformer {
    type Err;
    type Value;

    /// Transform a value.
    fn transform(&self, x: Self::Value) -> Self::Value;

    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit(examples: impl Iterator<Item = Self::Value> + Clone) -> Result<Self, Self::Err>
    where
        Self: Sized;
}

pub trait TransformExt: Iterator {
    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit<T: Transformer<Value = Self::Item>>(self) -> Result<T, T::Err>
    where
        Self: Clone,
    {
        T::fit(self)
    }

    /// Maps the iterator with a transformer.
    ///
    /// This only transforms the iterator based on the provided transformer,
    /// not on the iterator itself.
    fn transform_by<T: Transformer>(self, transformer: T) -> Transformed<Self, T>
    where
        Self: Sized,
    {
        Transformed::new(self, transformer)
    }

    /// Fits a transformer from the elements of the iterator
    /// and then maps the iterator with the transformer.
    fn fit_transform<T: Transformer<Value = Self::Item>>(
        self,
    ) -> Result<Transformed<Self, T>, T::Err>
    where
        Self: Clone,
    {
        let t = T::fit(self.clone())?;
        Ok(self.transform_by(t))
    }
}
impl<I: Iterator> TransformExt for I {}

#[derive(Debug, Clone)]
pub struct Transformed<I, T> {
    iter: I,
    transformer: T,
}
impl<I, T> Iterator for Transformed<I, T>
where
    I: Iterator,
    T: Transformer<Value = I::Item>,
{
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
