use clap::{Parser, ValueEnum};
use std::str::FromStr;
use tetris_atlas::tetris::{Column, Rotation};
use tetris_atlas::{
    set_global_threadpool, tetris_evolution_player_model, tetris_exceed_the_mean,
    tetris_simple_player_model, tetris_transition_model, tetris_transition_transformer_model,
    tetris_world_model,
};
use time::OffsetDateTime;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum TrainModel {
    Evolution,
    SimpleGoalPolicy,
    ExceedTheMean,
    Transition,
    TransitionTransformer,
    WorldGoalPolicy,
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

        #[arg(long, value_enum, help = "Select which training routine to run")]
        model: TrainModel,
    },
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[arg(long, global = true, help = "Enable Chrome tracing")]
    trace: bool,

    #[arg(
        long,
        global = true,
        help = "Directory to save Chrome trace files (only used with --trace)"
    )]
    trace_dir: Option<String>,

    #[arg(
        long,
        global = true,
        help = "Path to Perfetto configuration file (only used with --trace)"
    )]
    perfetto_config: Option<String>,

    #[arg(
        long,
        global = true,
        help = "Parse arguments and exit immediately (for validation)"
    )]
    noop: bool,

    #[command(subcommand)]
    command: Commands,
}

fn main() {
    info!("Starting tetris-atlas");
    set_global_threadpool();

    let cli = Cli::parse();

    // If noop flag is set, exit immediately after parsing arguments
    if cli.noop {
        println!("✓ Arguments parsed successfully (--noop mode)");
        std::process::exit(0);
    }

    let verbosity = cli.verbose.saturating_add(2).clamp(0, 5);
    let level = Level::from_str(verbosity.to_string().as_str()).unwrap();

    let registry = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::filter::LevelFilter::from_level(level));

    // Initialize logging with optional Perfetto tracing
    let _guard: Option<()> = if cli.trace {
        let _timestamp = OffsetDateTime::now_utc()
            .format(
                &time::format_description::parse("[year][month][day]_[hour][minute][second]")
                    .unwrap(),
            )
            .unwrap();
        registry.init();
        None
    } else {
        registry.init();
        info!("Logging initialized at level: {}", level);
        None
    };

    match &cli.command {
        Commands::Train {
            logdir,
            checkpoint_dir,
            run_name,
            model,
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
            match model {
                TrainModel::Evolution => {
                    tetris_evolution_player_model::train_population(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::SimpleGoalPolicy => {
                    tetris_simple_player_model::train_goal_policy(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::ExceedTheMean => {
                    tetris_exceed_the_mean::train_exceed_the_mean_policy(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::Transition => {
                    tetris_transition_model::train_game_transition_model(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::TransitionTransformer => {
                    tetris_transition_transformer_model::train_game_transition_model(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::WorldGoalPolicy => {
                    tetris_world_model::train_goal_policy(run_name, logdir, checkpoint_dir)
                        .unwrap();
                }
            }
        }
        Commands::Play {} => {
            use std::io::Write;
            use tetris_atlas::tetris::TetrisGame;

            let mut game = TetrisGame::new();
            println!("Welcome to Tetris! (Type 'quit' to exit)");
            println!("\nInstructions:");
            println!("  - Enter the column (0-9) and rotation (0-3) separated by space");
            println!("  - Example: '5 2' places the piece at column 5 with rotation 2");
            println!("  - Type 'quit' or 'exit' to stop playing\n");

            loop {
                // Print current game state
                println!("\n{}", "=".repeat(50));
                println!("Current Piece: {}", game.current_piece());
                println!("Lines Cleared: {}", game.lines_cleared);
                println!("Pieces Played: {}", game.piece_count);
                println!("\nCurrent Board:");
                println!("{}", game.board);

                // Get all valid placements for the current piece
                let valid_placements = game.current_placements();

                if valid_placements.is_empty() {
                    println!("\n🎮 Game Over! No valid moves available.");
                    println!("Final Score:");
                    println!("  Lines Cleared: {}", game.lines_cleared);
                    println!("  Pieces Played: {}", game.piece_count);
                    break;
                }

                // Show available valid moves
                println!("\nValid moves for piece '{}':", game.current_piece());
                let mut move_list = Vec::new();
                for placement in valid_placements {
                    // Extract the raw values from the Display output
                    let col_str = format!("{}", placement.orientation.column);
                    let rot_str = format!("{}", placement.orientation.rotation);
                    // Parse the column value from "Column(X)" format
                    let col_val = col_str
                        .trim_start_matches("Column(")
                        .trim_end_matches(")")
                        .to_string();
                    move_list.push((col_val, rot_str));
                }
                // Sort and deduplicate for cleaner display
                move_list.sort();
                move_list.dedup();

                println!("  (column, rotation) pairs:");
                for (col, rot) in &move_list {
                    print!("  ({}, {}) ", col, rot);
                }
                println!("\n");

                // Get user input
                print!("Enter your move (column rotation): ");
                std::io::stdout().flush().expect("Failed to flush stdout");

                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");

                let input = input.trim();
                if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
                    println!("\nThanks for playing!");
                    println!("Final Score:");
                    println!("  Lines Cleared: {}", game.lines_cleared);
                    println!("  Pieces Played: {}", game.piece_count);
                    break;
                }

                // Parse input as column and rotation
                let parts: Vec<&str> = input.split_whitespace().collect();
                if parts.len() != 2 {
                    println!("❌ Invalid input. Please enter two numbers: column rotation");
                    println!("   Example: '5 2' for column 5, rotation 2");
                    continue;
                }

                // Parse column
                let column_value = match parts[0].parse::<u8>() {
                    Ok(c) => c,
                    Err(_) => {
                        println!("❌ Invalid column. Please enter a number between 0 and 9.");
                        continue;
                    }
                };

                // Parse rotation
                let rotation_value = match parts[1].parse::<u8>() {
                    Ok(r) => r,
                    Err(_) => {
                        println!("❌ Invalid rotation. Please enter a number between 0 and 3.");
                        continue;
                    }
                };

                // Validate column range
                if column_value >= Column::MAX {
                    println!(
                        "❌ Invalid column {}. Must be between 0 and {}.",
                        column_value,
                        Column::MAX - 1
                    );
                    continue;
                }

                // Validate rotation range
                if rotation_value >= 4 {
                    println!(
                        "❌ Invalid rotation {}. Must be between 0 and 3.",
                        rotation_value
                    );
                    continue;
                }

                // Create the placement and check if it's valid for the current piece
                // Use unsafe transmute like the codebase does internally
                let column: Column = unsafe { std::mem::transmute(column_value) };
                let rotation: Rotation = unsafe { std::mem::transmute(rotation_value) };

                // Find matching valid placement
                let matching_placement = valid_placements
                    .iter()
                    .find(|p| p.orientation.column == column && p.orientation.rotation == rotation);

                match matching_placement {
                    Some(placement) => {
                        println!(
                            "\n✓ Playing: {} at column {} with rotation {}",
                            game.current_piece(),
                            column_value,
                            rotation_value
                        );

                        let is_lost = game.apply_placement(*placement);

                        if is_lost.into() {
                            println!("\n{}", game.board);
                            println!("\n🎮 Game Over! The piece couldn't fit on the board.");
                            println!("Final Score:");
                            println!("  Lines Cleared: {}", game.lines_cleared);
                            println!("  Pieces Played: {}", game.piece_count);
                            break;
                        }
                    }
                    None => {
                        println!(
                            "❌ Invalid move! Column {} with rotation {} is not a valid placement for piece '{}'.",
                            column_value,
                            rotation_value,
                            game.current_piece()
                        );
                        println!("   Please choose from the valid moves listed above.");
                        continue;
                    }
                }
            }
        }
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
