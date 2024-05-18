use thiserror::Error;

pub mod distance;
pub mod linear_regression;
pub mod mean;
pub mod standard_deviation;

#[derive(Debug, Error, Clone, Copy)]
#[error("Empty sequence")]
pub struct EmptySequenceError;
