use clap::Parser;
use itertools::Itertools;
use rand::{rng, seq::IteratorRandom};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    hash::{DefaultHasher, Hash, Hasher, SipHasher},
    ops::AddAssign,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tetris_atlas::{
    atlas::{Atlas, AtlasNode, Dfs, TetrisState},
    tetris_board::{BitSetter, Countable, Shiftable, TetrisBoard},
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
    Run {
        #[arg(long, help = "Path to save the atlas file")]
        save_file: Option<String>,

        #[arg(long, default_value = "8", help = "Maximum depth for the search")]
        max_depth: usize,
    },
    Explore {
        #[arg(long, help = "Path to the atlas file")]
        atlas_file: String,
    },
    Play {
        #[arg(long, default_value = "16", help = "Number of pieces to play")]
        num_pieces: usize,
    },
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
        Commands::Play { num_pieces } => {
            let mut rng = rng();
            let mut state = TetrisState::default();
            println!("Board: {}", state.board);
            for _ in 0..*num_pieces {
                let placements = state.all_next_placements();
                let (new_bag, new_placement) = placements.choose(&mut rng).unwrap();
                state.board.play_piece(
                    new_placement.piece,
                    new_placement.rotation,
                    new_placement.column.0,
                );
                state.bag = new_bag;

                println!("Board: {}", state.board);
                println!("Bag: {}", new_bag);
                println!("Placements: {}", new_placement);

                // Wait for user to press Enter before continuing
                println!("Press Enter to continue...");
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
            }
        }
        Commands::Explore { atlas_file } => {
            let atlas = Atlas::load_atlas(atlas_file);
            atlas.interactive_traverse();
        }

        Commands::Run {
            max_depth,
            save_file,
        } => {
            let max_depth: Option<usize> = Some(*max_depth);
            let root: AtlasNode = AtlasNode {
                state: TetrisState::default(),
            };

            let dfs = Dfs::new(root, max_depth);

            let count_counts = 16;
            let running_count_counts = (0..count_counts)
                .map(|_| Arc::new(Mutex::new(0)))
                .collect::<Vec<_>>();

            // Launch a background thread to print progress every second
            let also_visited = dfs.visited.clone();
            let also_atlas = dfs.atlas.clone();
            let also_running_count_counts = running_count_counts.clone();
            let progress_running = Arc::new(AtomicBool::new(true));
            let also_progress_running = progress_running.clone();
            let handle = std::thread::spawn(move || {
                while also_progress_running.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(3));

                    let total_visited = also_visited.read().unwrap().len();
                    println!("Total nodes visited: {}", total_visited);

                    let total = also_running_count_counts
                        .iter()
                        .map(|mutex| *mutex.lock().unwrap())
                        .sum::<usize>();
                    println!("Total nodes processed: {}", total);

                    // get the total size of the atlas
                    let atlas_size = also_atlas.read().unwrap().inner.len();
                    println!("Atlas size: {}", atlas_size);
                }
            });

            dfs.clone().into_par_iter().for_each(|n| {
                running_count_counts[n.node.state.board.play_board.count() % count_counts]
                    .lock()
                    .unwrap()
                    .add_assign(1);
            });

            // Signal the thread to exit and wait for it to finish
            progress_running.store(false, Ordering::Relaxed);
            handle.join().unwrap();

            // print total counters
            let mut total_count = 0;
            for (i, count) in running_count_counts.iter().enumerate() {
                println!("Counter {}: {}", i, *count.lock().unwrap());
                total_count += *count.lock().unwrap();
            }
            println!("Total count: {}", total_count);

            if let Some(save_file) = save_file {
                dfs.atlas.write().unwrap().save_atlas(save_file);
            }

            // calc stats of the atlas
            let also_dfs = dfs.clone();
            let atlas = also_dfs.atlas.read().unwrap();

            let atlas_size = atlas.inner.len();
            println!("Atlas size: {}", atlas_size);
            let num_unique_boards = atlas.inner.iter().map(|e| e.board).unique().count();
            println!("Number of unique boards: {}", num_unique_boards);
        }
    }
}
