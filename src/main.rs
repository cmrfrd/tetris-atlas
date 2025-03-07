use clap::Parser;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{str::FromStr, sync::Mutex};
use tetris_atlas::{
    atlas::{AtlasNode, Dfs, TetrisState},
    tetris_board::{BitSetter, Shiftable, TetrisBoard},
};
use tracing::{Level, info};

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
            let root = AtlasNode {
                state: TetrisState::default(),
            };

            let dfs = Dfs::<AtlasNode>::new(root, 10).into_par_iter();
            let running_count = Mutex::new(0);
            let print_every = 100_000;
            dfs.for_each(|_| {
                let mut count = running_count.lock().unwrap();
                *count += 1;
                if *count % print_every == 0 {
                    println!("{}", count);
                    // println!("{}", node.state.board);
                    // Request input from user
                    // println!("Just hit enter to continue...");
                    // let mut input = String::new();
                    // std::io::stdin()
                    //     .read_line(&mut input)
                    //     .expect("Failed to read line");
                }
            });
        }
    }
}
