use std::path::PathBuf;

use clap::Args;

#[derive(Debug, Clone, Args)]
pub struct LinRegrArgs {
    pub input: PathBuf,
    pub x: Vec<String>,
    pub y: String,
}
impl LinRegrArgs {
    pub fn run(self) -> anyhow::Result<()> {
        todo!()
    }
}
