//! Deterministic, codec-independent legal-play inputs for codec benchmarks.

use tetris_game::{
    TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement,
};

/// Number of raw boards in each fixed codec workload.
pub(crate) const SAMPLES: usize = 256;

/// A narrow full-bag game that exercises two-byte IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SixColumnTetris;

impl TetrisGameConfig for SixColumnTetris {
    const ROWS: usize = 20;
    const COLS: usize = 6;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
}

/// Collects 256 legal boards without consulting codec IDs or admission rules.
///
/// Every history respects the height and hole caps and consumes the configured bag.
/// A blocked draw starts a new history; previously collected witnesses remain valid.
pub(crate) fn legal_rows<C: TetrisGameConfig, const R: usize, const H: usize>() -> Vec<[u16; R]>
where
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    let mut seed = 0xb04d_c0de_0002u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let mut board = TetrisBoard::<{ C::COLS }, { C::ROWS }>::new();
    let mut bag = C::PIECE_SET;
    let mut rows = vec![[0; R]];
    for _ in 0..65_536 {
        if rows.len() == SAMPLES {
            break;
        }

        if bag.empty() {
            bag = C::PIECE_SET;
        }

        let pieces: Vec<_> = TetrisPiece::all()
            .into_iter()
            .filter(|&p| bag.contains(p))
            .collect();

        let piece =
            pieces[usize::try_from(next() % pieces.len() as u64).expect("piece index fits usize")];

        let candidates: Vec<_> = TetrisPiecePlacement::<{ C::COLS }>::ALL_PLACEMENTS
            .iter()
            .filter(|p| p.piece == piece)
            .filter_map(|&p| {
                let mut candidate = board;
                candidate.apply_piece_placement(p);
                (!candidate.is_lost()
                    && candidate.height() as usize <= R
                    && candidate.total_holes() as usize <= H)
                    .then_some(candidate)
            })
            .collect();

        if candidates.is_empty() {
            board = TetrisBoard::new();
            bag = C::PIECE_SET;
            continue;
        }

        board = candidates
            [usize::try_from(next() % candidates.len() as u64).expect("fixture index fits usize")];
        bag.remove(piece);

        rows.push(std::array::from_fn(|y| {
            board
                .as_limbs()
                .iter()
                .enumerate()
                .fold(0u16, |row, (x, col)| {
                    row | (u16::try_from((col >> y) & 1).expect("one bit fits u16") << x)
                })
        }));
    }

    assert_eq!(
        rows.len(),
        SAMPLES,
        "bounded seed must supply enough legal fixtures"
    );
    rows
}
