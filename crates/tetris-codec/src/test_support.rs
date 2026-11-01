//! Fixtures shared by module-local codec tests.
use tetris_game::{TetrisGameConfig, TetrisPieceBagState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Game<const COLS: usize, const ROWS: usize>;
impl<const COLS: usize, const ROWS: usize> TetrisGameConfig for Game<COLS, ROWS> {
    const ROWS: usize = ROWS;
    const COLS: usize = COLS;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::new();
}

use std::sync::LazyLock;
use tetris_game::{StandardTetris, TetrisBoard, TetrisPiecePlacement};
struct Footprint {
    placement: TetrisPiecePlacement,
    columns: [u32; 10],
    height: usize,
}
static FOOTPRINTS: LazyLock<Vec<Footprint>> = LazyLock::new(|| {
    TetrisPiecePlacement::ALL_PLACEMENTS
        .into_iter()
        .map(|placement| {
            let mut b = TetrisBoard::<StandardTetris>::new();
            b.apply_piece_placement(placement);
            Footprint {
                placement,
                columns: b.as_limbs(),
                height: b.height() as usize,
            }
        })
        .collect()
});

/// Independent engine oracle. Tests use widths <= the engine's placement inventory.
pub(crate) fn has_predecessor<C: TetrisGameConfig, const R: usize>(rows: &[u16; R]) -> bool
where
    [(); C::COLS]:,
{
    let columns =
        std::array::from_fn(|x| (0..R).fold(0u32, |v, y| v | (u32::from((rows[y] >> x) & 1) << y)));
    let target = TetrisBoard::<C>::from_limbs(columns);
    for shape in FOOTPRINTS.iter() {
        if shape.height > R
            || shape
                .columns
                .iter()
                .enumerate()
                .any(|(x, &v)| x >= C::COLS && v != 0)
        {
            continue;
        }
        for dy in 0..=R - shape.height {
            if !(0..C::COLS)
                .all(|x| (shape.columns[x] << dy) & columns[x] == shape.columns[x] << dy)
            {
                continue;
            }
            let mut before = TetrisBoard::<C>::from_limbs(std::array::from_fn(|x| {
                columns[x] & !(shape.columns[x] << dy)
            }));
            let cols = before.as_limbs();
            if (0..before.height()).any(|y| cols.iter().all(|c| c & (1 << y) == 0)) {
                continue;
            }
            let result = before.apply_piece_placement(shape.placement);
            if result.lines_cleared == 0 && before == target {
                return true;
            }
        }
    }
    false
}
pub(crate) fn admitted<C: TetrisGameConfig, const R: usize, const H: usize>(rows: &[u16; R]) -> bool
where
    [(); C::COLS]:,
{
    let full = (1u16 << C::COLS) - 1;
    if rows.iter().any(|&r| r & !full != 0) {
        return false;
    }
    let height = rows.iter().rposition(|&r| r != 0).map_or(0, |y| y + 1);
    if rows[..height].iter().any(|&r| r == 0 || r == full) {
        return false;
    }
    let (mut divisor, mut width) = (4, C::COLS);
    while width != 0 {
        (divisor, width) = (width, divisor % width);
    }
    if rows.iter().map(|r| r.count_ones() as usize).sum::<usize>() % divisor != 0 {
        return false;
    }
    let mut holes = 0;
    for x in 0..C::COLS {
        if let Some(top) = rows.iter().rposition(|r| r & (1 << x) != 0) {
            holes += rows[..top].iter().filter(|r| **r & (1 << x) == 0).count();
        }
    }
    if holes > H {
        return false;
    }
    if C::COLS >= 4 && height == R && R >= 2 {
        let mut unsupported = 0;
        let mut x = 0;
        while x < C::COLS {
            if rows[R - 1] & (1 << x) == 0 {
                x += 1;
                continue;
            }
            let start = x;
            let mut supported = false;
            while x < C::COLS && rows[R - 1] & (1 << x) != 0 {
                supported |= rows[R - 2] & (1 << x) != 0;
                x += 1;
            }
            if !supported {
                unsupported += 1;
                if unsupported > 1 || x - start > 3 {
                    return false;
                }
            }
        }
    }
    !(height == R
        && R <= 28
        && C::COLS > 4
        && (rows[R - 1] & !(rows[R - 1] << 1)).count_ones() > 1
        && !has_predecessor::<C, R>(rows))
}
