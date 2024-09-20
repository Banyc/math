use std::path::PathBuf;

use banyc_polars_util::read_df_file;
use clap::Args;
use math::{
    statistics::lin_regr::{adjusted_r_squared, t_test_params, LinearRegressionEstimator, Sample},
    transformer::Estimate,
};
use polars::prelude::*;
use primitive::iter::{AssertIteratorItemExt, VecZip};
use strict_num::FiniteF64;

#[derive(Debug, Clone, Args)]
pub struct LinRegrArgs {
    pub input: PathBuf,
    #[clap(short, long)]
    pub x: Vec<String>,
    #[clap(short, long)]
    pub y: String,
}
impl LinRegrArgs {
    pub fn run(self) -> anyhow::Result<()> {
        let df = read_df_file(self.input)?;

        let mut columns = vec![];
        columns.extend(self.x.iter().map(col));
        columns.push(col(&self.y));

        let df = df.select(&columns).collect()?;

        let mut columns = vec![];
        let y_column = df
            .column(&self.y)?
            .f64()?
            .iter()
            .collect::<Vec<Option<f64>>>();
        columns.push(y_column.into_iter());
        for x in &self.x {
            let column = df.column(x)?.f64()?.iter().collect::<Vec<Option<f64>>>();
            columns.push(column.into_iter());
        }

        let zip = VecZip::new(columns);

        let examples = zip
            .filter_map(|columns| {
                columns
                    .into_iter()
                    .assert_item::<Option<f64>>()
                    .map(|column| column.and_then(FiniteF64::new))
                    .assert_item::<Option<FiniteF64>>()
                    .collect::<Option<Vec<FiniteF64>>>()
            })
            .map(|columns| {
                let mut columns = columns.into_iter().assert_item::<FiniteF64>();
                let y = columns.next().unwrap();
                let x = columns.collect::<Vec<FiniteF64>>();
                Example { x, y }
            })
            .collect::<Vec<Example>>();

        let estimator = LinearRegressionEstimator;
        let model = estimator.fit(examples.iter())?;

        let adjusted_r_squared = adjusted_r_squared(&model, examples.iter())?;
        println!("adjusted R-squared: {adjusted_r_squared}");

        let t_test_params = t_test_params(&model, examples.iter())?;
        let p_values = t_test_params.two_sided_p_values();
        let p_values = p_values.iter().map(|p| p.get()).collect::<Vec<f64>>();

        println!("residual SE: {}", t_test_params.residual_standard_error);
        println!("df: {}", t_test_params.df);
        println!("b_0: {}", model.slopes()[0]);
        println!("slopes: {:?}", &model.slopes()[1..]);
        println!("slope SE: {:?}", t_test_params.slope_standard_errors);
        println!("t: {:?}", t_test_params.t);
        println!("p values: {p_values:?}");

        Ok(())
    }
}

struct Example {
    pub x: Vec<FiniteF64>,
    pub y: FiniteF64,
}
impl Sample for &Example {
    fn predictors(&self) -> impl Iterator<Item = FiniteF64> + Clone {
        self.x.iter().copied()
    }

    fn response(&self) -> FiniteF64 {
        self.y
    }
}
