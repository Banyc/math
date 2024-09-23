use core::num::NonZeroUsize;
use std::collections::HashMap;

use primitive::iter::AssertIteratorItemExt;
use strict_num::FiniteF64;
use thiserror::Error;

use crate::{
    matrix::{Container2D, Index, MatrixBuf, Size, VecMatrix},
    statistics::variance::VarianceExt,
    transformer::Estimate,
};

use super::{standard_deviation::StandardDeviationExt, EmptySequenceError};

pub trait Sample {
    fn predictors(&self) -> impl Iterator<Item = FiniteF64> + Clone;
    fn response(&self) -> FiniteF64;
}

#[derive(Debug, Clone)]
pub struct LinearRegressionEstimator;
impl<V> Estimate<V> for LinearRegressionEstimator
where
    V: Sample,
{
    type Err = ExamplesError;
    type Output = LinearRegressionBuf;

    #[allow(non_snake_case)]
    fn fit(&self, examples: impl Iterator<Item = V> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let XTX_inv = XTX_inv(examples.clone())?;

        let k: NonZeroUsize = num_predictor_variables(examples.clone())?;

        let y_sum: f64 = examples
            .clone()
            .map(|example| example.response().get())
            .sum();
        let mut x_y_sums: Vec<f64> = vec![];
        for i in 0..k.get() {
            let sum: f64 = examples
                .clone()
                .map(|example| {
                    example.predictors().nth(i).unwrap().get() * example.response().get()
                })
                .sum();
            x_y_sums.push(sum);
        }

        let mut XTy_data = vec![];
        XTy_data.push(y_sum);
        XTy_data.extend(x_y_sums.iter().copied());
        let XTy_rows = NonZeroUsize::new(k.get() + 1).unwrap();
        let size = Size {
            rows: XTy_rows,
            cols: NonZeroUsize::new(1).unwrap(),
        };
        let XTy = MatrixBuf::new(size, XTy_data);

        let b = XTX_inv.mul_matrix(&XTy);

        let mut slopes = vec![];
        for row in 0..b.size().rows.get() {
            let slope = b.get(Index { row, col: 0 });
            slopes.push(slope);
        }
        Ok(LinearRegressionBuf::new(slopes))
    }
}

fn num_predictor_variables<V>(
    examples: impl Iterator<Item = V> + Clone,
) -> Result<NonZeroUsize, ExamplesError>
where
    V: Sample,
{
    let first: V = examples
        .clone()
        .next()
        .ok_or(ExamplesError::EmptyExamples)?;
    let k: usize = first.predictors().count();
    for example in examples.clone() {
        if example.predictors().count() != k {
            return Err(ExamplesError::NumPredictorsNotConstant);
        }
    }
    Ok(NonZeroUsize::new(k).unwrap())
}
#[derive(Debug, Error, Clone, Copy)]
pub enum ExamplesError {
    #[error("number of predictors is not constant")]
    NumPredictorsNotConstant,
    #[error("no examples")]
    EmptyExamples,
}

#[allow(non_snake_case)]
fn XTX_inv<V>(examples: impl Iterator<Item = V> + Clone) -> Result<VecMatrix<f64>, ExamplesError>
where
    V: Sample,
{
    let k: NonZeroUsize = num_predictor_variables(examples.clone())?;
    let n: usize = examples.clone().count();

    let mut x_sums: Vec<f64> = vec![];
    for i in 0..k.get() {
        let sum: f64 = examples
            .clone()
            .map(|example| example.predictors().nth(i).unwrap().get())
            .sum();
        x_sums.push(sum);
    }
    let mut x_x_sums: HashMap<(usize, usize), f64> = HashMap::new();
    for first in 0..k.get() {
        for second in 0..k.get() {
            if second < first {
                continue;
            }

            let sum: f64 = examples
                .clone()
                .map(|example| {
                    example.predictors().nth(first).unwrap().get()
                        * example.predictors().nth(second).unwrap().get()
                })
                .sum();
            x_x_sums.insert((first, second), sum);
        }
    }

    let XTX_rows = NonZeroUsize::new(k.get() + 1).unwrap();
    let mut XTX_data = vec![];
    for first in 0..XTX_rows.get() {
        for second in 0..XTX_rows.get() {
            if first == 0 && second == 0 {
                XTX_data.push(n as f64);
                continue;
            }
            let min = first.min(second);
            let max = first.max(second);
            if min == 0 {
                let i = max - 1;
                XTX_data.push(x_sums[i]);
                continue;
            }
            let min = min - 1;
            let max = max - 1;
            XTX_data.push(*x_x_sums.get(&(min, max)).unwrap());
        }
    }
    let size = Size {
        rows: XTX_rows,
        cols: XTX_rows,
    };
    let XTX = MatrixBuf::new(size, XTX_data);
    Ok(XTX.inverse())
}

#[derive(Debug, Clone)]
pub struct LinearRegressionBuf {
    /// ```math
    /// (b_0, b_1, ..., b_k)
    /// ```
    ///
    /// - $k$: the number of predictor variables
    slopes: Vec<f64>,
}
impl LinearRegressionBuf {
    pub fn new(slopes: Vec<f64>) -> Self {
        assert!(1 < slopes.len());
        Self { slopes }
    }
}
impl LinearRegression for LinearRegressionBuf {
    fn predict(
        &self,
        predictors: impl Iterator<Item = f64> + Clone,
    ) -> Result<f64, LinearRegressionError> {
        if predictors.clone().count() + 1 != self.slopes.len() {
            return Err(LinearRegressionError::NumPredictorsNumSlopesMismatched);
        }

        let mut sum = self.slopes[0];
        for (slope, predictor) in self.slopes[1..].iter().copied().zip(predictors) {
            sum += slope * predictor
        }

        Ok(sum)
    }
    fn slopes(&self) -> &[f64] {
        &self.slopes
    }
}

pub trait LinearRegression {
    fn predict(
        &self,
        predictors: impl Iterator<Item = f64> + Clone,
    ) -> Result<f64, LinearRegressionError>;
    fn slopes(&self) -> &[f64];
}
#[derive(Debug, Error, Clone, Copy)]
pub enum LinearRegressionError {
    #[error("number of predictors and slopes are not the same")]
    NumPredictorsNumSlopesMismatched,
}

pub trait LinearRegressionExt: LinearRegression {
    fn adjusted_r_squared<V>(
        &self,
        examples: impl Iterator<Item = V> + Clone,
    ) -> Result<f64, EmptySequenceError>
    where
        V: Sample,
    {
        let n = examples.clone().count();
        let k = self.slopes()[1..].len();
        let adjustment = (n - 1) as f64 / (n - k - 1) as f64;
        Ok(1. - self.standard_residuals_variance(examples)? * adjustment)
    }
    fn r_squared<V>(
        &self,
        examples: impl Iterator<Item = V> + Clone,
    ) -> Result<f64, EmptySequenceError>
    where
        V: Sample,
    {
        Ok(1. - self.standard_residuals_variance(examples)?)
    }
    fn standard_residuals_variance<V>(
        &self,
        examples: impl Iterator<Item = V> + Clone,
    ) -> Result<f64, EmptySequenceError>
    where
        V: Sample,
    {
        let responses = examples.clone().map(|example| example.response().get());
        let s2_y = responses.clone().variance()?;
        let residuals = self.residuals(examples.clone());
        let s2_res = residuals.iter().copied().variance()?;
        Ok(s2_res / s2_y)
    }
    fn residuals<V>(&self, examples: impl Iterator<Item = V> + Clone) -> Vec<f64>
    where
        V: Sample,
    {
        let responses = examples.clone().map(|example| example.response().get());
        let predicted_responses = examples
            .clone()
            .map(|example| self.predict(example.predictors().map(|x| x.get())).unwrap())
            .assert_item::<f64>();
        predicted_responses
            .zip(responses.clone())
            .map(|(y_hat, y)| y - y_hat)
            .collect::<Vec<f64>>()
    }

    /// Null hypothesis: $b_i = 0$
    ///
    /// Exclude the intercept.
    #[allow(non_snake_case)]
    fn t_test_params<V>(
        &self,
        examples: impl Iterator<Item = V> + Clone,
    ) -> Result<TTestParams, TTestParamsError>
    where
        V: Sample,
    {
        let k = self.slopes()[1..].len();
        let n = examples.clone().count();
        let residuals = self.residuals(examples.clone());
        let residual_standard_error: f64 = residuals.iter().copied().standard_deviation().unwrap();

        let XTX_inv = XTX_inv(examples.clone()).map_err(TTestParamsError::Examples)?;
        let mut slope_standard_errors = vec![];
        for i in 1..=k {
            let index = Index { row: i, col: i };
            let value = XTX_inv.get(index);
            let se = residual_standard_error * value.sqrt();
            slope_standard_errors.push(se);
        }
        let t_values: Vec<f64> = self.slopes()[1..]
            .iter()
            .copied()
            .zip(slope_standard_errors.iter().copied())
            .map(|(b, se)| (b - 0.) / se)
            .collect::<Vec<f64>>();
        let df = n - k - 1;
        let df = NonZeroUsize::new(df).ok_or(TTestParamsError::TooFewExamples)?;
        Ok(TTestParams {
            residual_standard_error,
            slope_standard_errors,
            t: t_values,
            df,
        })
    }
}
impl<T> LinearRegressionExt for T where T: LinearRegression {}

#[derive(Debug, Clone)]
pub struct TTestParams {
    pub residual_standard_error: f64,
    pub slope_standard_errors: Vec<f64>,
    pub t: Vec<f64>,
    pub df: NonZeroUsize,
}
#[derive(Debug, Error, Clone, Copy)]
pub enum TTestParamsError {
    #[error("too few examples")]
    TooFewExamples,
    #[error("{0}")]
    Examples(ExamplesError),
}

#[cfg(test)]
mod tests {
    use primitive::float::FloatExt;

    use super::*;

    pub struct TheSample {
        pub x: Vec<FiniteF64>,
        pub y: FiniteF64,
    }
    impl Sample for &TheSample {
        fn predictors(&self) -> impl Iterator<Item = FiniteF64> + Clone {
            self.x.iter().copied()
        }

        fn response(&self) -> FiniteF64 {
            self.y
        }
    }

    #[test]
    fn test_fit_1() {
        let samples = [(vec![0.], 0.), (vec![1.], 1.)];
        let samples = samples
            .into_iter()
            .map(|(x, y)| TheSample {
                x: x.into_iter()
                    .map(FiniteF64::new)
                    .collect::<Option<Vec<FiniteF64>>>()
                    .unwrap(),
                y: FiniteF64::new(y).unwrap(),
            })
            .collect::<Vec<TheSample>>();
        let estimator = LinearRegressionEstimator;
        let model = estimator.fit(samples.iter()).unwrap();
        println!("{model:?}");
        assert!(model.slopes[0].closes_to(0.));
        assert!(model.slopes[1].closes_to(1.));

        let r_squared = model.r_squared(samples.iter()).unwrap();
        println!("R-squared: {r_squared}");
    }

    #[test]
    fn test_fit_2() {
        let samples = [
            (vec![1.21], 1.69),
            (vec![3.], 5.89),
            (vec![5.16], 4.11),
            (vec![8.31], 5.49),
            (vec![10.21], 8.65),
        ];
        let samples = samples
            .iter()
            .map(|(x, y)| TheSample {
                x: x.iter()
                    .copied()
                    .map(FiniteF64::new)
                    .collect::<Option<Vec<FiniteF64>>>()
                    .unwrap(),
                y: FiniteF64::new(*y).unwrap(),
            })
            .collect::<Vec<TheSample>>();
        let estimator = LinearRegressionEstimator;
        let model = estimator.fit(samples.iter()).unwrap();
        println!("{model:?}");
        assert!(f64::abs(model.slopes[0] - 2.034) < 0.001);
        assert!(f64::abs(model.slopes[1] - 0.5615) < 0.001);

        let r_squared = model.r_squared(samples.iter()).unwrap();
        println!("R-squared: {r_squared}");
        let adjusted_r_squared = model.adjusted_r_squared(samples.iter()).unwrap();
        println!("adjusted R-squared: {adjusted_r_squared}");
        let t_test_params = model.t_test_params(samples.iter()).unwrap();
        println!("{t_test_params:?}");
        // println!(
        //     "two-sided p values: {:?}",
        //     t_test_params.two_sided_p_values()
        // );
    }

    /// ref: <https://ecampusontario.pressbooks.pub/introstats/chapter/13-3-standard-error-of-the-estimate/>
    #[test]
    fn test_fit_3() {
        let samples = [
            (vec![3., 23., 60.], 4.),
            (vec![8., 32., 114.], 5.),
            (vec![9., 28., 45.], 2.),
            (vec![4., 60., 187.], 6.),
            (vec![3., 62., 175.], 7.),
            (vec![1., 43., 125.], 8.),
            (vec![6., 60., 93.], 7.),
            (vec![3., 37., 57.], 3.),
            (vec![2., 24., 47.], 5.),
            (vec![5., 64., 128.], 5.),
            (vec![2., 28., 66.], 7.),
            (vec![1., 66., 146.], 8.),
            (vec![7., 35., 89.], 5.),
            (vec![5., 37., 56.], 2.),
            (vec![0., 59., 65.], 4.),
            (vec![2., 32., 95.], 6.),
            (vec![6., 76., 82.], 5.),
            (vec![5., 25., 90.], 7.),
            (vec![0., 55., 137.], 9.),
            (vec![3., 34., 91.], 8.),
            (vec![5., 54., 184.], 7.),
            (vec![1., 57., 60.], 9.),
            (vec![0., 68., 39.], 7.),
            (vec![2., 66., 187.], 10.),
            (vec![0., 50., 49.], 5.),
        ];
        let samples = samples
            .iter()
            .map(|(x, y)| TheSample {
                x: x.iter()
                    .copied()
                    .map(FiniteF64::new)
                    .collect::<Option<Vec<FiniteF64>>>()
                    .unwrap(),
                y: FiniteF64::new(*y).unwrap(),
            })
            .collect::<Vec<TheSample>>();
        let estimator = LinearRegressionEstimator;
        let model = estimator.fit(samples.iter()).unwrap();
        println!("{model:?}");
        assert!(f64::abs(model.slopes[0] - 4.7993) < 0.001);
        assert!(f64::abs(model.slopes[1] - -0.3818) < 0.001);
        assert!(f64::abs(model.slopes[2] - 0.0046) < 0.001);
        assert!(f64::abs(model.slopes[3] - 0.0233) < 0.001);

        let r_squared = model.r_squared(samples.iter()).unwrap();
        assert!(r_squared.closes_to(0.506629665));
        let adjusted_r_squared = model.adjusted_r_squared(samples.iter()).unwrap();
        assert!(adjusted_r_squared.closes_to(0.436148189));
        let t_test_params = model.t_test_params(samples.iter()).unwrap();
        assert_eq!(t_test_params.df.get(), 21);
        println!("{t_test_params:?}");
        // println!(
        //     "two-sided p values: {:?}",
        //     t_test_params.two_sided_p_values()
        // );
    }
}
