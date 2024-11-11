use std::path::PathBuf;

use banyc_polars_util::read_df_file;
use clap::Args;
use math::{
    statistics::lin_regr::{
        LinearRegression, LinearRegressionEstimator, LinearRegressionExt, Sample, TTestParams,
    },
    transformer::Estimate,
};
use polars::prelude::*;
use primitive::{
    iter::{assertion::AssertIteratorItemExt, vec_zip::VecZip},
    ops::float::{UnitR, R},
};

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
                    .map(|column| column.and_then(R::new))
                    .assert_item::<Option<R<f64>>>()
                    .collect::<Option<Vec<R<f64>>>>()
            })
            .map(|columns| {
                let mut columns = columns.into_iter().assert_item::<R<f64>>();
                let y = columns.next().unwrap();
                let x = columns.collect::<Vec<R<f64>>>();
                Example { x, y }
            })
            .collect::<Vec<Example>>();

        let estimator = LinearRegressionEstimator;
        let model = estimator.fit(examples.iter())?;

        let adjusted_r_squared = model.adjusted_r_squared(examples.iter())?;
        println!("adjusted R-squared: {adjusted_r_squared}");

        let t_test_params = model.t_test_params(examples.iter())?;
        let p_values = two_sided_p_values(&t_test_params);
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
    pub x: Vec<R<f64>>,
    pub y: R<f64>,
}
impl Sample for &Example {
    fn predictors(&self) -> impl Iterator<Item = R<f64>> + Clone {
        self.x.iter().copied()
    }
    fn response(&self) -> R<f64> {
        self.y
    }
}

fn two_sided_p_values(params: &TTestParams) -> Vec<UnitR<f64>> {
    let t = params
        .t
        .iter()
        .copied()
        .map(R::new)
        .collect::<Option<Vec<R<f64>>>>()
        .unwrap();
    t.iter()
        .copied()
        .map(|t| {
            let t = statistical_inference::R::new(t.get()).unwrap();
            let p = statistical_inference::distributions::t::T_SCORE_TABLE
                .p_value_two_sided(params.df, t);
            UnitR::new(p.get()).unwrap()
        })
        .collect::<Vec<UnitR<f64>>>()
}
