use clap::Parser;
use std::str::FromStr;
use tetris_atlas::train;
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

// #[derive(Clone)]
// struct Stats {
//     timer: Instant,
//     counters: Vec<Arc<Mutex<usize>>>,
//     visited: Arc<DashSet<AtlasNode>>,
//     atlas: Arc<RwLock<Atlas>>,
// }

// impl Stats {
//     fn new(
//         count_counts: usize,
//         visited: Arc<DashSet<AtlasNode>>,
//         atlas: Arc<RwLock<Atlas>>,
//     ) -> Self {
//         let counters = (0..count_counts).map(|_| Arc::new(Mutex::new(0))).collect();
//         Self {
//             timer: Instant::now(),
//             counters,
//             visited,
//             atlas,
//         }
//     }

//     fn update(&self, node: &AtlasNode) {
//         let index = node.board.play_board.count() % self.counters.len();
//         let mut counter = self.counters[index].lock().unwrap();
//         *counter += 1;
//     }

//     fn print_stats(&self) {
//         let mut ascii_table = AsciiTable::default();
//         ascii_table.set_max_width(40);
//         ascii_table
//             .column(0)
//             .set_header("Stat Item")
//             .set_align(Align::Left);
//         ascii_table
//             .column(1)
//             .set_header("Value")
//             .set_align(Align::Right);
//         let mut data: Vec<Vec<String>> = Vec::with_capacity(2_usize.pow(8));

//         let elapsed = self.timer.elapsed();
//         let duration_str = format!(
//             "{}:{:02}:{:02}",
//             elapsed.as_secs() / 3600,
//             (elapsed.as_secs() % 3600) / 60,
//             elapsed.as_secs() % 60
//         );
//         data.push(vec!["Duration".to_string(), duration_str]);

//         let total: usize = self
//             .counters
//             .iter()
//             .map(|counter| *counter.lock().unwrap())
//             .sum();
//         data.push(vec![
//             "Total Nodes Counted".to_string(),
//             total.to_formatted_string(&Locale::en),
//         ]);

//         let visited_size = self.visited.len();
//         data.push(vec![
//             "Unique Visited Nodes".to_string(),
//             visited_size.to_formatted_string(&Locale::en),
//         ]);

//         let atlas_size = self.atlas.read().unwrap().inner.len();
//         data.push(vec![
//             "Atlas Size".to_string(),
//             atlas_size.to_formatted_string(&Locale::en),
//         ]);

//         let display_data: Vec<Vec<&dyn Display>> = data
//             .iter()
//             .map(|row| row.iter().map(|s| s as &dyn Display).collect())
//             .collect();
//         ascii_table.print(display_data);
//     }
// }

#[derive(Debug, Parser)]
enum Commands {
    Run {
        #[arg(long, help = "Path to save the atlas file")]
        save_file: Option<String>,

        #[arg(long, default_value = "8", help = "Maximum depth for the search")]
        max_depth: usize,
    },
    Explore {
        #[arg(short = 'a', long = "atlas-file", help = "Path to the atlas file")]
        atlas_file: String,
    },
    Play {},
    Test {},
    Train {
        #[arg(long, help = "Path to save the tensorboard logs")]
        logdir: Option<String>,

        #[arg(long, help = "Path to load/save model checkpoints")]
        checkpoint_dir: Option<String>,

        #[arg(long, help = "Name of the training run")]
        run_name: String,
    },
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[arg(long, global = true, help = "Enable Chrome tracing")]
    trace: bool,

    #[command(subcommand)]
    command: Commands,
}

fn main() {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let cli = Cli::parse();
    let filter = setup_logging(cli.verbose);
    info!("Debug level: level={}", filter);

    let _guard = if cli.trace {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match &cli.command {
        Commands::Train {
            logdir,
            checkpoint_dir,
            run_name,
        } => {
            let ulid = ulid::Ulid::new().to_string();
            let run_name = format!("{run_name}_{ulid}");
            let logdir = logdir.as_ref().map(|s| {
                let path = std::path::Path::new(s).join(&run_name);
                std::fs::create_dir_all(&path).expect("Failed to create log directory");
                path
            });
            let checkpoint_dir = checkpoint_dir.as_ref().map(|s| {
                let path = std::path::Path::new(s).join(&run_name);
                std::fs::create_dir_all(&path).expect("Failed to create checkpoint directory");
                path
            });
            // train::train();
            train::train_game_transformer(logdir.clone());
        }
        // Commands::Play {} => {
        //     let mut current_node = AtlasNode::default();
        //     println!("Welcome to Tetris! (Type 'quit' to exit)");

        //     loop {
        //         // Print current board state
        //         println!("\nCurrent Board:");
        //         println!("{}", current_node.board);
        //         println!("Current Bag: {}", current_node.bag);

        //         // Get all possible next moves
        //         let next_nodes = current_node.children().collect::<Vec<_>>();
        //         if next_nodes.is_empty() {
        //             println!("Game over! No more valid moves.");
        //             break;
        //         }

        //         // Display available information
        //         println!("\nCurrent bag: {}", current_node.bag);
        //         println!("Enter piece, rotation, and column e.g. 1 0 0 (or 'quit' to exit)");

        //         // Get user input
        //         let mut input = String::new();
        //         std::io::stdin()
        //             .read_line(&mut input)
        //             .expect("Failed to read line");

        //         let input = input.trim();
        //         if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
        //             println!("Thanks for playing!");
        //             break;
        //         }

        //         // Parse input as piece, rotation, column
        //         let parts: Vec<&str> = input.split_whitespace().collect();
        //         if parts.len() != 3 {
        //             println!("Invalid input. Please enter three numbers: piece rotation column");
        //             continue;
        //         }

        //         // Parse each part
        //         let piece = match parts[0].parse::<usize>() {
        //             Ok(p) => p,
        //             Err(_) => {
        //                 println!("Invalid piece number. Please enter a number.");
        //                 continue;
        //             }
        //         };

        //         let rotation = match parts[1].parse::<usize>() {
        //             Ok(r) => r,
        //             Err(_) => {
        //                 println!("Invalid rotation number. Please enter a number.");
        //                 continue;
        //             }
        //         };

        //         let column = match parts[2].parse::<usize>() {
        //             Ok(c) => c,
        //             Err(_) => {
        //                 println!("Invalid column number. Please enter a number.");
        //                 continue;
        //             }
        //         };

        //         // Find matching placement
        //         let next_move = next_nodes.iter().find(|(_, placement)| {
        //             placement.piece.0 == piece as u8
        //                 && placement.rotation.0 == rotation as u8
        //                 && placement.column.0 == column as u8
        //         });

        //         match next_move {
        //             Some((new_node, placement)) => {
        //                 println!(
        //                     "\nPlaying: Piece {}, Rotation {}, Column {}",
        //                     placement.piece, placement.rotation, placement.column
        //                 );
        //                 current_node = new_node.clone();
        //             }
        //             None => {
        //                 println!("Invalid move. This placement is not possible.");
        //                 continue;
        //             }
        //         }
        //     }
        // }
        // Commands::Explore { atlas_file } => {
        //     let atlas = Atlas::load_atlas(atlas_file);
        //     atlas.interactive_traverse();
        // }

        // Commands::Run {
        //     max_depth,
        //     save_file,
        // } => {
        //     let max_depth: Option<usize> = Some(*max_depth);
        //     let root: AtlasNode = AtlasNode::default();
        //     let atlas_search = AtlasSearch::new(root, max_depth);

        //     let stats = Stats::new(64, atlas_search.visited.clone(), atlas_search.atlas.clone());

        //     // Launch a background thread to print progress every second
        //     let also_stats = stats.clone();
        //     let progress_running = Arc::new(AtomicBool::new(true));
        //     let also_progress_running = progress_running.clone();
        //     let handle = std::thread::spawn(move || {
        //         while also_progress_running.load(Ordering::Relaxed) {
        //             std::thread::sleep(std::time::Duration::from_secs(3));
        //             also_stats.print_stats();
        //         }
        //     });

        //     let also_stats = stats.clone();
        //     atlas_search.clone().into_par_iter().for_each(|n| {
        //         also_stats.update(&n.node);
        //     });

        //     // Signal the thread to exit and wait for it to finish
        //     progress_running.store(false, Ordering::Relaxed);
        //     handle.join().unwrap();

        //     if let Some(save_file) = save_file {
        //         atlas_search.atlas.write().unwrap().save_atlas(save_file);
        //     }
        // }
        _ => unreachable!(),
    }
}
