use clap::Parser;
use std::str::FromStr;
use tracing::{Level, info};

use tetris_atlas::tetris_board::{BitSetter, Shiftable, TetrisBoard};

fn setup_logging(verbosity: u8) -> String {
    let verbosity = verbosity.saturating_add(2).clamp(0, 5);
    let level = Level::from_str(verbosity.to_string().as_str()).unwrap();
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();
    level.to_string()
}

#[derive(Debug, Parser)]
enum Commands {
    Run,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

fn main() {
    let cli = Cli::parse();
    let filter = setup_logging(cli.verbose);
    info!("Debug level: level={}", filter);
    match &cli.command {
        Commands::Run => {
            let mut board = TetrisBoard::default();
            info!("{}", board);
        }
    }
}
