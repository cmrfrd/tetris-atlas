use clap::{Parser, ValueEnum};
use std::str::FromStr;
use tetris_atlas::tetris::TetrisGame;
use tetris_atlas::tetris_beam_supervised;
use tetris_atlas::tetris_policy_gradients;
use tetris_atlas::{
    set_global_threadpool, tetris_dqn, tetris_evolution_player_model,
    tetris_q_learning_transformer, tetris_simple_imitation, tetris_simple_player_model,
    tetris_transition_model, tetris_transition_transformer_model, tetris_tui,
    tetris_value_function, tetris_world_model,
};
use time::OffsetDateTime;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum TrainModel {
    Dqn,
    Evolution,
    SimpleGoalPolicy,
    PolicyGradients,
    BeamSupervised,
    SimpleImitation,
    ValueFunction,
    QLearningTransformer,
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

        #[arg(long, help = "Resume training from latest checkpoint")]
        resume: bool,
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
            resume,
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
                TrainModel::Dqn => {
                    tetris_dqn::train_tetris_dqn(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                        *resume,
                    )
                    .unwrap();
                }
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
                TrainModel::PolicyGradients => {
                    tetris_policy_gradients::train_tetris_policy_gradients(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::BeamSupervised => {
                    tetris_beam_supervised::train_tetris_beam_supervised(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                        *resume,
                    )
                    .unwrap();
                }
                TrainModel::SimpleImitation => {
                    tetris_simple_imitation::train_simple_imitation_policy(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                    )
                    .unwrap();
                }
                TrainModel::ValueFunction => {
                    tetris_value_function::train_value_function_policy(
                        run_name.clone(),
                        logdir.clone(),
                        checkpoint_dir.clone(),
                        *resume,
                    )
                    .unwrap();
                }
                TrainModel::QLearningTransformer => {
                    tetris_q_learning_transformer::train_q_learning_policy(
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
            tetris_tui::run().expect("Failed to run TUI");
        }
        Commands::Test {} => {
            let mut game = TetrisGame::new();
            println!("{}", game);

            let placements = game.current_placements();
            let placement = placements[0];
            println!("{}", placement);

            let result = game.apply_placement(placement);
            println!(
                "IsLost: {}, LinesCleared: {}",
                result.is_lost, result.lines_cleared
            );
            println!("{}", game);
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
