pub mod mean_imputer;
pub mod min_max_scaler;
pub mod proportion_scaler;
pub mod standard_scaler;

pub trait Estimate {
    type Err;
    type Value;
    type Output;

    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit(
        &self,
        examples: impl Iterator<Item = Self::Value> + Clone,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: Sized;
}
pub trait EstimateExt: Iterator {
    /// Fits a transformer from the elements of the iterator,
    /// so that you can use this transformer to transform another iterator.
    fn fit<E: Estimate<Value = Self::Item>>(self, estimator: &E) -> Result<E::Output, E::Err>
    where
        Self: Clone,
    {
        estimator.fit(self)
    }
}
impl<I: Iterator> EstimateExt for I {}

pub trait Transform {
    type Err;
    type Value;

    /// Transform a value.
    fn transform(&self, x: Self::Value) -> Result<Self::Value, Self::Err>;
}
pub trait TransformExt: Iterator {
    /// Maps the iterator with a transformer.
    ///
    /// This only transforms the iterator based on the provided transformer,
    /// not on the iterator itself.
    fn transform_by<T: Transform>(self, transformer: T) -> Transformed<Self, T>
    where
        Self: Sized,
    {
        Transformed::new(self, transformer)
    }

    /// Fits a transformer from the elements of the iterator
    /// and then maps the iterator with the transformer.
    fn fit_transform<
        E: Estimate<Value = Self::Item, Output = T>,
        T: Transform<Value = Self::Item>,
    >(
        self,
        estimator: &E,
    ) -> Result<Transformed<Self, T>, E::Err>
    where
        Self: Clone,
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
impl<I, T> Iterator for Transformed<I, T>
where
    I: Iterator,
    T: Transform<Value = I::Item>,
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
