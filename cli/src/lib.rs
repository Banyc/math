use clap::{Parser, Subcommand};
use lin_regr::LinRegrArgs;

pub mod lin_regr;

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}
impl Cli {
    pub fn run(self) -> anyhow::Result<()> {
        self.command.run()
    }
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    LinRegr(LinRegrArgs),
}
impl Command {
    pub fn run(self) -> anyhow::Result<()> {
        match self {
            Command::LinRegr(args) => args.run(),
        }
    }
}
