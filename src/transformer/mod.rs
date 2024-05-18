pub mod mean_imputer;
pub mod min_max_scaler;
pub mod proportion_scaler;
pub mod standard_scaler;

/// `V`: type of the items in the input iterator
pub trait Estimate<V> {
    type Err;
    type Output;

    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit(&self, examples: impl Iterator<Item = V> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized;
}
pub trait EstimateExt {
    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit<V, E: Estimate<V>>(self, estimator: &E) -> Result<E::Output, E::Err>
    where
        Self: Iterator<Item = V> + Clone,
    {
        estimator.fit(self)
    }
}
impl<I: Iterator> EstimateExt for I {}

/// `V`: type of the input to be transformed
pub trait Transform<V> {
    type Err;

    /// Transform a value.
    fn transform(&self, x: V) -> Result<V, Self::Err>;
}
pub trait TransformExt {
    /// Maps the iterator with a transformer.
    ///
    /// This only transforms the iterator based on the provided transformer,
    /// not on the iterator itself.
    fn transform_by<V, T: Transform<V>>(self, transformer: T) -> Transformed<Self, T>
    where
        Self: Sized,
    {
        Transformed::new(self, transformer)
    }

    /// Fits a transformer from the elements of the iterator
    /// and then maps the iterator with the transformer.
    fn fit_transform<V, E: Estimate<V, Output = T>, T: Transform<V>>(
        self,
        estimator: &E,
    ) -> Result<Transformed<Self, T>, E::Err>
    where
        Self: Iterator<Item = V> + Clone,
    {
        let t = estimator.fit(self.clone())?;
        Ok(self.transform_by(t))
    }
}
impl<I: Iterator> TransformExt for I {}

#[derive(Debug, Clone)]
pub struct Transformed<I, T> {
    iter: I,
    transformer: T,
}
impl<V, I, T> Iterator for Transformed<I, T>
where
    I: Iterator<Item = V>,
    T: Transform<V>,
{
    type Item = Result<I::Item, T::Err>;

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
