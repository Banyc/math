use std::num::NonZeroUsize;

use strict_num::FiniteF64;
use thiserror::Error;

use crate::{
    lin_alg::{Index, Matrix},
    statistics::variance::VarianceExt,
    transformer::Estimate,
};

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
    type Err = LinearRegressionEstimatorError;
    type Output = LinearRegression;

    #[allow(non_snake_case)]
    fn fit(&self, examples: impl Iterator<Item = V> + Clone) -> Result<Self::Output, Self::Err>
    where
        Self: Sized,
    {
        let first: V = examples
            .clone()
            .next()
            .ok_or(LinearRegressionEstimatorError::EmptyExamples)?;
        let k: usize = first.predictors().count();
        for example in examples.clone() {
            if example.predictors().count() != k {
                return Err(LinearRegressionEstimatorError::NumPredictorsNotConstant);
            }
        }

        let n: usize = examples.clone().count();

        let mut x_sums: Vec<f64> = vec![];
        for i in 0..k {
            let sum: f64 = examples
                .clone()
                .map(|example| example.predictors().nth(i).unwrap().get())
                .sum();
            x_sums.push(sum);
        }
        let mut x_x_sums: Vec<Vec<f64>> = vec![];
        for first in 0..k {
            let mut row = vec![];
            for second in 0..k {
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
                row.push(sum);
            }
            x_x_sums.push(row);
        }

        let y_sum: f64 = examples
            .clone()
            .map(|example| example.response().get())
            .sum();
        let mut x_y_sums: Vec<f64> = vec![];
        for i in 0..k {
            let sum: f64 = examples
                .clone()
                .map(|example| {
                    example.predictors().nth(i).unwrap().get() * example.response().get()
                })
                .sum();
            x_y_sums.push(sum);
        }

        let XTX_rows = NonZeroUsize::new(k + 1).unwrap();
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
                XTX_data.push(x_x_sums[min][max]);
            }
        }
        let XTX = Matrix::new(XTX_rows, XTX_data);
        let XTX_inv = XTX.inverse();

        let mut XTy_data = vec![];
        XTy_data.push(y_sum);
        XTy_data.extend(x_y_sums.iter().copied());
        let XTy_rows = NonZeroUsize::new(k + 1).unwrap();
        let XTy = Matrix::new(XTy_rows, XTy_data);

        let b = XTX_inv.mul_matrix(&XTy);

        let mut slopes = vec![];
        for row in 0..b.rows().get() {
            let slope = b.cell(Index { row, col: 0 });
            slopes.push(slope);
        }
        Ok(LinearRegression::new(slopes))
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum LinearRegressionEstimatorError {
    #[error("number of predictors is not constant")]
    NumPredictorsNotConstant,
    #[error("no examples")]
    EmptyExamples,
}

#[derive(Debug, Clone)]
pub struct LinearRegression {
    slopes: Vec<f64>,
}
impl LinearRegression {
    pub fn new(slopes: Vec<f64>) -> Self {
        assert!(1 < slopes.len());
        Self { slopes }
    }

    pub fn predict(
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

    pub fn num_slopes(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.slopes.len()).unwrap()
    }
}
#[derive(Debug, Error, Clone, Copy)]
pub enum LinearRegressionError {
    #[error("number of predictors and slopes are not the same")]
    NumPredictorsNumSlopesMismatched,
}

pub fn adjusted_r_squared<V>(
    model: &LinearRegression,
    examples: impl Iterator<Item = V> + Clone,
) -> Result<f64, RSquareError>
where
    V: Sample,
{
    let n = examples.clone().count();
    let k = model.num_slopes().get();
    let adjustment = (n - 1) as f64 / (n - k - 1) as f64;
    Ok(1. - standard_s_res(model, examples)? * adjustment)
}
pub fn r_squared<V>(
    model: &LinearRegression,
    examples: impl Iterator<Item = V> + Clone,
) -> Result<f64, RSquareError>
where
    V: Sample,
{
    Ok(1. - standard_s_res(model, examples)?)
}
fn standard_s_res<V>(
    model: &LinearRegression,
    examples: impl Iterator<Item = V> + Clone,
) -> Result<f64, RSquareError>
where
    V: Sample,
{
    let responses = examples.clone().map(|example| example.response().get());
    let s2_y = responses
        .clone()
        .variance()
        .map_err(|_| RSquareError::EmptyExamples)?;
    let predicted_responses: Vec<f64> = examples
        .clone()
        .map(|example| model.predict(example.predictors().map(|x| x.get())))
        .collect::<Result<Vec<f64>, _>>()
        .unwrap();
    let residuals = predicted_responses
        .iter()
        .copied()
        .zip(responses.clone())
        .map(|(y_hat, y)| y - y_hat);
    let s2_res = residuals
        .clone()
        .variance()
        .map_err(|_| RSquareError::EmptyExamples)?;
    Ok(s2_res / s2_y)
}
#[derive(Debug, Error, Clone, Copy)]
pub enum RSquareError {
    #[error("no examples")]
    EmptyExamples,
    #[error("{0}")]
    LinearRegression(LinearRegressionError),
}

#[cfg(test)]
mod tests {
    use crate::float::FloatExt;

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

        let r_squared = r_squared(&model, samples.iter()).unwrap();
        println!("R-squared: {r_squared}");
        let adjusted_r_squared = adjusted_r_squared(&model, samples.iter()).unwrap();
        println!("adjusted R-squared: {adjusted_r_squared}");
    }
}
