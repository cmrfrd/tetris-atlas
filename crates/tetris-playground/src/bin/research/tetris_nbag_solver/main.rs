mod search;

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use tetris_game::{TetrisBoard, TetrisGameRng, TetrisPiece, TetrisPieceBag};

use search::{SearchOutcome, search};

#[derive(Debug, Parser)]
#[command(name = "tetris_nbag_solver")]
#[command(about = "Find a placement sequence over N random bags that returns the board to empty")]
struct Cli {
    #[arg(long, default_value_t = 5)]
    num_bags: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 8)]
    width: usize,

    #[arg(long, default_value_t = usize::MAX)]
    max_nodes: usize,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    #[arg(long, default_value_t = false)]
    replay: bool,
}

fn generate_sequence(num_bags: u32, seed: u64) -> Vec<TetrisPiece> {
    let mut rng = TetrisGameRng::new(seed);
    let mut seq = Vec::with_capacity((num_bags * 7) as usize);
    for _ in 0..num_bags {
        let mut bag = TetrisPieceBag::new_ordered();
        bag.shuffle(&mut rng);
        for _ in 0..7 {
            seq.push(bag.rand_next(&mut rng));
        }
    }
    seq
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.num_bags % 5 != 0 {
        eprintln!(
            "WARNING: num_bags={} is not a multiple of 5. \
             Empty-to-empty requires 28*N cells = 10*k lines, so N must be divisible by 5.",
            cli.num_bags
        );
    }

    let sequence = generate_sequence(cli.num_bags, cli.seed);
    let total_pieces = sequence.len();

    println!("=== tetris_nbag_solver ===");
    println!(
        "num_bags={}, seed={}, width={}, max_nodes={}",
        cli.num_bags, cli.seed, cli.width, cli.max_nodes
    );
    println!("total_pieces={}", total_pieces);
    print!("sequence: ");
    for (i, piece) in sequence.iter().enumerate() {
        if i > 0 && i % 7 == 0 {
            print!("| ");
        }
        print!("{piece} ");
    }
    println!();
    println!();

    let target = TetrisBoard::new();
    let start = Instant::now();
    let outcome = search(&sequence, target, cli.width, cli.max_nodes, cli.verbose);
    let elapsed = start.elapsed();

    match outcome {
        SearchOutcome::Found(result) => {
            println!("FOUND solution in {elapsed:.2?}");
            println!(
                "  nodes_expanded={}, lines_cleared={}",
                result.nodes_expanded, result.lines_cleared
            );
            println!("  final_board == empty: {}", result.final_board == target);

            println!();
            println!("Placement sequence:");
            for (i, p) in result.placements.iter().enumerate() {
                println!("  step {:3}: piece={} placement={}", i, sequence[i], p);
            }

            if cli.replay {
                println!();
                println!("=== Replay ===");
                let mut board = TetrisBoard::new();
                for (i, p) in result.placements.iter().enumerate() {
                    let outcome = board.apply_piece_placement(*p);
                    println!(
                        "step {:3}: {} -> lines_cleared={} lost={}",
                        i, p, outcome.lines_cleared, outcome.is_lost
                    );
                    println!("{board}");
                }
            }
        }
        SearchOutcome::Exhausted { nodes_expanded } => {
            println!("EXHAUSTED search space in {elapsed:.2?}");
            println!("  nodes_expanded={nodes_expanded}");
            println!("  No empty-to-empty solution found with these parameters.");
            println!("  Try increasing --width or --max-nodes.");
        }
    }

    Ok(())
}
