#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable

import numpy as np


ROWS = 20
COLS = 10
BOARD_SIZE = ROWS * COLS
COUNT_STATES = COLS + 1
Q_SIZE = ROWS * COUNT_STATES
ALIVE_INDEX = BOARD_SIZE + Q_SIZE
DEAD_INDEX = ALIVE_INDEX + 1
HOMOG_INDEX = DEAD_INDEX + 1
STATE_SIZE = HOMOG_INDEX + 1
CANONICAL_ACTION_COUNT = 117
RUST_PIECE_ORDER = ("O", "I", "S", "Z", "T", "L", "J")


@dataclass(frozen=True)
class ProjectionBlock:
    out_offset: int
    in_offset: int
    block: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class BlockProjectionMatrix:
    shape: tuple[int, int]
    blocks: tuple[ProjectionBlock, ...] = field(repr=False)

    def matvec(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.int64)
        if arr.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},)")

        out = np.zeros(self.shape[0], dtype=np.int64)
        for projection_block in self.blocks:
            block = projection_block.block
            out_start = projection_block.out_offset
            in_start = projection_block.in_offset
            out[out_start : out_start + block.shape[0]] += (
                block @ arr[in_start : in_start + block.shape[1]]
            )
        return out

    def left_multiply(self, left: np.ndarray | Iterable[int]) -> np.ndarray:
        left_arr = np.asarray(left, dtype=np.int64)
        if left_arr.ndim != 2 or left_arr.shape[1] != self.shape[0]:
            raise ValueError(f"left matrix must have {self.shape[0]} columns")

        out = np.zeros((left_arr.shape[0], self.shape[1]), dtype=np.int64)
        for projection_block in self.blocks:
            block = projection_block.block
            out_start = projection_block.out_offset
            in_start = projection_block.in_offset
            out[:, in_start : in_start + block.shape[1]] += (
                left_arr[:, out_start : out_start + block.shape[0]] @ block
            )
        return out

    def __matmul__(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        return self.matvec(vector)


@dataclass(frozen=True)
class ComposedFeatureStepMatrix:
    left: np.ndarray = field(repr=False)
    right: BlockProjectionMatrix = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.right.shape[1]

    def matvec(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        projected = self.right @ vector
        return np.asarray(self.left, dtype=np.int64) @ projected

    def __matmul__(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        return self.matvec(vector)


@dataclass(frozen=True)
class SparseLinearMatrix:
    shape: tuple[int, int]
    rows: tuple[dict[int, int], ...] = field(repr=False)

    def matvec(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.int64)
        if arr.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},)")

        out = np.zeros(self.shape[0], dtype=np.int64)
        for row_idx, row in enumerate(self.rows):
            total = 0
            for col_idx, coeff in row.items():
                total += coeff * int(arr[col_idx])
            out[row_idx] = total
        return out

    def __matmul__(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        return self.matvec(vector)


@dataclass(frozen=True)
class MonomialBasis:
    monomials: tuple[tuple[int, ...], ...]
    index: dict[tuple[int, ...], int] = field(repr=False)

    @classmethod
    def from_coordinates(cls, coordinates: Iterable[int], max_degree: int) -> MonomialBasis:
        coords = tuple(sorted(set(coordinates)))
        if max_degree < 0:
            raise ValueError("max_degree must be nonnegative")

        monomials: list[tuple[int, ...]] = [()]
        for degree in range(1, max_degree + 1):
            monomials.extend(tuple(combo) for combo in combinations(coords, degree))
        return cls(
            monomials=tuple(monomials),
            index={monomial: idx for idx, monomial in enumerate(monomials)},
        )

    @property
    def size(self) -> int:
        return len(self.monomials)


@dataclass(frozen=True)
class PersistentZTransitionResult:
    basis: MonomialBasis
    z_state: np.ndarray = field(repr=False)
    transition_matrix: SparseLinearMatrix = field(repr=False)
    z_next: np.ndarray = field(repr=False)
    compact_next: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class BranchSpec:
    label: str
    guard: dict[tuple[int, ...], int] = field(repr=False)
    matrix: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ActionMatrixBranch:
    label: str
    y: int
    clear_rows: tuple[int, ...]
    is_death: bool = False

    def dense_matrix(self, placement: Placement) -> np.ndarray:
        if self.is_death:
            return death_matrix()

        mask = piece_mask(placement.piece, placement.rotation, placement.column, self.y)
        return clear_matrix(self.clear_rows) @ lock_matrix(mask)


@dataclass(frozen=True)
class BranchActionMatrix:
    placement: Placement
    branches: tuple[ActionMatrixBranch, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return STATE_SIZE, len(self.branches) * STATE_SIZE

    def matvec(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.int64)
        if arr.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},)")

        out = np.zeros(STATE_SIZE, dtype=np.int64)
        for branch_idx, branch in enumerate(self.branches):
            start = branch_idx * STATE_SIZE
            chunk = arr[start : start + STATE_SIZE]
            if not np.any(chunk):
                continue
            out += branch.dense_matrix(self.placement).astype(np.int64) @ chunk
        return out

    def __matmul__(self, vector: np.ndarray | Iterable[int]) -> np.ndarray:
        return self.matvec(vector)


@dataclass(frozen=True)
class ActionMatrixFamily:
    placements: tuple[Placement, ...]
    matrices: tuple[BranchActionMatrix, ...]

    def __len__(self) -> int:
        return len(self.matrices)

    def matrix_for(self, placement: Placement) -> BranchActionMatrix:
        for matrix in self.matrices:
            if matrix.placement == placement:
                return matrix
        raise KeyError(f"placement {placement!r} is not in this action family")


@dataclass(frozen=True)
class ClosureReport:
    action_count: int
    branch_count: int
    processed_features: int
    feature_count: int
    max_degree: int
    hit_feature_cap: bool
    hit_degree_cap: bool
    degree_histogram: dict[int, int]


# Coordinates are (row, col) with row 0 as the floor. Rotation 0 is chosen to
# make the demo readable; this is a matrix model demonstrator, not an SRS kick
# implementation.
BASE_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "L": ((0, 0), (0, 1), (0, 2), (1, 2)),
    "J": ((0, 0), (0, 1), (0, 2), (1, 0)),
}


@dataclass(frozen=True)
class Placement:
    piece: str
    rotation: int
    column: int


@dataclass(frozen=True)
class QuadraticProductLiftResult:
    base_state: np.ndarray = field(repr=False)
    quadratic_state: np.ndarray = field(repr=False)
    product_projection_matrix: np.ndarray = field(repr=False)
    product_state: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class GuardedCopyLiftResult:
    selector: np.ndarray = field(repr=False)
    source_state: np.ndarray = field(repr=False)
    product_state: np.ndarray = field(repr=False)
    lifted_state: np.ndarray = field(repr=False)
    projection_matrix: np.ndarray = field(repr=False)
    guarded_state: np.ndarray = field(repr=False)
    product_offset: int
    quadratic_lift: QuadraticProductLiftResult | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CollisionOccupancyLiftResult:
    cells: tuple[int, ...]
    lifted_state: np.ndarray = field(repr=False)
    projection_matrix: np.ndarray = field(repr=False)
    local_occupancy_tensor: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LinearizedCollisionResult:
    cells: tuple[int, ...]
    selector: np.ndarray = field(repr=False)
    local_occupancy_tensor: np.ndarray = field(repr=False)
    selector_projection: np.ndarray = field(repr=False)
    occupancy_lift: CollisionOccupancyLiftResult = field(repr=False)


@dataclass(frozen=True)
class PlacementSelectorLiftResult:
    collision_selector: np.ndarray = field(repr=False)
    clear_selector: np.ndarray = field(repr=False)
    base_state: np.ndarray = field(repr=False)
    quadratic_state: np.ndarray = field(repr=False)
    product_projection_matrix: np.ndarray = field(repr=False)
    product_state: np.ndarray = field(repr=False)
    selector_projection_matrix: np.ndarray = field(repr=False)
    selector: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LinearizedLockResult:
    next_state: np.ndarray = field(repr=False)
    selector: np.ndarray = field(repr=False)
    guarded_state: np.ndarray = field(repr=False)
    guarded_lift: GuardedCopyLiftResult = field(repr=False)
    branch_matrix: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LocalCountTensorLiftResult:
    rows: tuple[int, ...]
    lifted_state: np.ndarray = field(repr=False)
    projection_matrix: np.ndarray = field(repr=False)
    local_count_tensor: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class PlacementLiftResult:
    next_state: np.ndarray = field(repr=False)
    selector: np.ndarray = field(repr=False)
    branch_labels: tuple[str, ...]
    guarded_state: np.ndarray = field(repr=False)
    guarded_lift: GuardedCopyLiftResult = field(repr=False)
    branch_matrix: np.ndarray = field(repr=False)
    lifted_state: np.ndarray = field(repr=False)
    state_matrix: np.ndarray = field(repr=False)
    guarded_state_offset: int
    selector_lift: PlacementSelectorLiftResult | None = field(default=None, repr=False)


@dataclass(frozen=True)
class LinearizedStepLiftResult:
    next_state: np.ndarray = field(repr=False)
    lifted_state: np.ndarray = field(repr=False)
    step_matrix: np.ndarray = field(repr=False)
    component_offsets: dict[str, int] = field(repr=False)
    guarded_state_offset: int
    guarded_state_size: int


@dataclass(frozen=True)
class LinearizedStepFeatureResult:
    next_state: np.ndarray = field(repr=False)
    feature_state: np.ndarray = field(repr=False)
    feature_to_step_matrix: BlockProjectionMatrix = field(repr=False)
    projected_step_lift: np.ndarray = field(repr=False)
    feature_step_matrix: ComposedFeatureStepMatrix = field(repr=False)
    feature_offsets: dict[str, int] = field(repr=False)


@dataclass(frozen=True)
class LinearizedHardDropResult:
    y_values: tuple[int, ...]
    y: int
    valid_vector: np.ndarray = field(repr=False)
    landing_vector: np.ndarray = field(repr=False)
    feature_state: np.ndarray = field(repr=False)
    selector_projection: np.ndarray = field(repr=False)
    selector: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class StepResult:
    next_state: np.ndarray = field(repr=False)
    y: int
    clear_rows: tuple[int, ...]
    lock_matrix: np.ndarray = field(repr=False)
    clear_matrix: np.ndarray = field(repr=False)
    linearized_lock: LinearizedLockResult = field(repr=False)
    linearized_collision: LinearizedCollisionResult = field(repr=False)
    linearized_clear: LinearizedClearResult = field(repr=False)
    placement_lift: PlacementLiftResult = field(repr=False)
    step_lift: LinearizedStepLiftResult = field(repr=False)
    step_feature: LinearizedStepFeatureResult = field(repr=False)
    linearized_hard_drop: LinearizedHardDropResult = field(repr=False)
    died: bool = False

    @property
    def branch_matrix(self) -> np.ndarray:
        return self.clear_matrix @ self.lock_matrix


@dataclass(frozen=True)
class LinearizedClearResult:
    next_state: np.ndarray = field(repr=False)
    candidate_rows: tuple[int, ...]
    clear_patterns: tuple[tuple[int, ...], ...]
    selector: np.ndarray = field(repr=False)
    local_count_tensor: np.ndarray = field(repr=False)
    count_lift: LocalCountTensorLiftResult = field(repr=False)
    selector_projection: np.ndarray = field(repr=False)
    guarded_state: np.ndarray = field(repr=False)
    guarded_lift: GuardedCopyLiftResult = field(repr=False)
    clear_lift_matrix: np.ndarray = field(repr=False)
    clear_lifted_state: np.ndarray = field(repr=False)
    clear_state_matrix: np.ndarray = field(repr=False)
    guarded_state_offset: int


def normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cell_list = list(cells)
    min_row = min(row for row, _ in cell_list)
    min_col = min(col for _, col in cell_list)
    return tuple(sorted((row - min_row, col - min_col) for row, col in cell_list))


def rotate_clockwise(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return normalize_cells((col, -row) for row, col in cells)


def all_rotations(piece: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    piece = piece.upper()
    if piece not in BASE_SHAPES:
        raise ValueError(f"unknown piece {piece!r}; expected one of {sorted(BASE_SHAPES)}")

    rotations: list[tuple[tuple[int, int], ...]] = []
    cells = normalize_cells(BASE_SHAPES[piece])
    for _ in range(4):
        if cells not in rotations:
            rotations.append(cells)
        cells = rotate_clockwise(cells)
    return tuple(rotations)


def shape_cells(piece: str, rotation: int) -> tuple[tuple[int, int], ...]:
    rotations = all_rotations(piece)
    return rotations[rotation % len(rotations)]


def as_board_vector(board: np.ndarray | Iterable[int]) -> np.ndarray:
    arr = np.asarray(board)
    if arr.shape == (ROWS, COLS):
        arr = arr.reshape(BOARD_SIZE)
    elif arr.shape != (BOARD_SIZE,):
        raise ValueError(f"board must have shape ({ROWS}, {COLS}) or ({BOARD_SIZE},)")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("board must be binary")
    return arr.astype(np.uint8, copy=False)


def as_state_vector(state: np.ndarray | Iterable[int]) -> np.ndarray:
    arr = np.asarray(state)
    if arr.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},)")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("state must be binary")
    if int(arr[ALIVE_INDEX]) + int(arr[DEAD_INDEX]) != 1:
        raise ValueError("exactly one of alive/dead must be set")
    if arr[DEAD_INDEX] == 1:
        expected = make_dead_state()
        if not np.array_equal(arr, expected):
            raise ValueError("dead states must be exactly the dead basis vector")
    elif arr[HOMOG_INDEX] != 1:
        raise ValueError("live states must have homogeneous coordinate 1")
    return arr.astype(np.uint8, copy=False)


def cell_index(row: int, col: int) -> int:
    return row * COLS + col


def board_grid(board: np.ndarray | Iterable[int]) -> np.ndarray:
    return as_board_vector(board).reshape(ROWS, COLS)


def row_sum_matrix() -> np.ndarray:
    matrix = np.zeros((ROWS, BOARD_SIZE), dtype=np.uint8)
    for row in range(ROWS):
        for col in range(COLS):
            matrix[row, cell_index(row, col)] = 1
    return matrix


def one_hot_counts_from_board(board: np.ndarray | Iterable[int]) -> np.ndarray:
    counts = row_sum_matrix() @ as_board_vector(board)
    q = np.zeros(Q_SIZE, dtype=np.uint8)
    q[np.arange(ROWS) * COUNT_STATES + counts] = 1
    return q


def empty_row_counts() -> np.ndarray:
    q = np.zeros(Q_SIZE, dtype=np.uint8)
    q[np.arange(ROWS) * COUNT_STATES] = 1
    return q


def counts_from_q(q: np.ndarray | Iterable[int]) -> np.ndarray:
    arr = np.asarray(q)
    if arr.shape != (Q_SIZE,):
        raise ValueError(f"q must have shape ({Q_SIZE},)")
    blocks = arr.reshape(ROWS, COUNT_STATES)
    if not np.all(blocks.sum(axis=1) == 1):
        raise ValueError("each row-count block must be one-hot")
    return blocks.argmax(axis=1)


def counts_from_q_or_dead(q: np.ndarray | Iterable[int]) -> np.ndarray:
    arr = np.asarray(q)
    if arr.shape != (Q_SIZE,):
        raise ValueError(f"q must have shape ({Q_SIZE},)")
    if not np.any(arr):
        return np.zeros(ROWS, dtype=np.int64)
    return counts_from_q(arr)


def make_state(board: np.ndarray | Iterable[int]) -> np.ndarray:
    board_vec = as_board_vector(board)
    q = one_hot_counts_from_board(board_vec)
    return np.concatenate([board_vec, q, np.array([1, 0, 1], dtype=np.uint8)])


def make_dead_state() -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=np.uint8)
    state[DEAD_INDEX] = 1
    return state


def board_from_state(state: np.ndarray | Iterable[int]) -> np.ndarray:
    return as_state_vector(state)[:BOARD_SIZE]


def q_from_state(state: np.ndarray | Iterable[int]) -> np.ndarray:
    return as_state_vector(state)[BOARD_SIZE : BOARD_SIZE + Q_SIZE]


def is_dead_state(state: np.ndarray | Iterable[int]) -> bool:
    return bool(as_state_vector(state)[DEAD_INDEX])


def is_alive_state(state: np.ndarray | Iterable[int]) -> bool:
    return bool(as_state_vector(state)[ALIVE_INDEX])


def piece_mask(piece: str, rotation: int, column: int, y: int) -> np.ndarray:
    mask = np.zeros(BOARD_SIZE, dtype=np.uint8)
    for drow, dcol in shape_cells(piece, rotation):
        row = y + drow
        col = column + dcol
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            raise ValueError(
                f"placement {(piece, rotation, column, y)!r} is outside the board"
            )
        mask[cell_index(row, col)] = 1
    return mask


def placement_y_range(piece: str, rotation: int, column: int) -> range:
    cells = shape_cells(piece, rotation)
    height = max(row for row, _ in cells) + 1
    width = max(col for _, col in cells) + 1
    if column < 0 or column + width > COLS:
        raise ValueError(f"placement {(piece, rotation, column)!r} is outside the board")
    return range(ROWS - height + 1)


def hard_drop_y(state: np.ndarray | Iterable[int], placement: Placement) -> int:
    board = board_from_state(state)
    for y in placement_y_range(placement.piece, placement.rotation, placement.column):
        mask = piece_mask(placement.piece, placement.rotation, placement.column, y)
        if placement_valid(board, mask):
            return y
    raise ValueError(f"placement {placement!r} has no legal hard-drop row")


def placement_valid(board: np.ndarray | Iterable[int], mask: np.ndarray | Iterable[int]) -> bool:
    board_vec = as_board_vector(board)
    mask_vec = as_board_vector(mask)
    return int(board_vec @ mask_vec) == 0


def hard_drop_selector_projection(y_count: int) -> np.ndarray:
    if y_count < 0:
        raise ValueError("y_count must be nonnegative")

    feature_size = 1 + 2 * y_count
    landing_offset = 1 + y_count
    projection = np.zeros((1 + y_count, feature_size), dtype=np.int64)
    projection[0, 0] = 1
    projection[0, landing_offset : landing_offset + y_count] = -1
    for idx in range(y_count):
        projection[1 + idx, landing_offset + idx] = 1
    return projection


def landing_vector_from_valid(valid_vector: np.ndarray | Iterable[int]) -> np.ndarray:
    valid = np.asarray(valid_vector, dtype=np.uint8)
    if valid.ndim != 1:
        raise ValueError("valid_vector must be one-dimensional")
    if not np.isin(valid, (0, 1)).all():
        raise ValueError("valid_vector must be binary")

    landing = np.zeros(valid.size, dtype=np.uint8)
    seen_lower_valid = False
    for idx, is_valid in enumerate(valid):
        if bool(is_valid) and not seen_lower_valid:
            landing[idx] = 1
        seen_lower_valid = seen_lower_valid or bool(is_valid)
    return landing


def dead_hard_drop_result(y_values: Iterable[int] = ()) -> LinearizedHardDropResult:
    y_tuple = tuple(int(y) for y in y_values)
    valid = np.zeros(len(y_tuple), dtype=np.uint8)
    landing = np.zeros(len(y_tuple), dtype=np.uint8)
    feature = np.concatenate([np.array([1], dtype=np.int64), valid, landing])
    projection = hard_drop_selector_projection(len(y_tuple))
    selector = projection @ feature
    if selector.tolist() != [1] + [0] * len(y_tuple):
        raise ValueError("dead hard-drop selector construction failed")
    return LinearizedHardDropResult(
        y_values=y_tuple,
        y=-1,
        valid_vector=valid,
        landing_vector=landing,
        feature_state=feature,
        selector_projection=projection,
        selector=selector.astype(np.uint8, copy=False),
    )


def linearized_hard_drop(
    state: np.ndarray | Iterable[int], placement: Placement
) -> LinearizedHardDropResult:
    x = as_state_vector(state)
    if is_dead_state(x):
        return dead_hard_drop_result()

    try:
        y_values = tuple(
            placement_y_range(placement.piece, placement.rotation, placement.column)
        )
    except ValueError:
        return dead_hard_drop_result()

    board = board_from_state(x)
    valid = np.zeros(len(y_values), dtype=np.uint8)
    for idx, y in enumerate(y_values):
        mask = piece_mask(placement.piece, placement.rotation, placement.column, y)
        valid[idx] = int(placement_valid(board, mask))

    landing = landing_vector_from_valid(valid)
    feature = np.concatenate([np.array([1], dtype=np.int64), valid, landing])
    projection = hard_drop_selector_projection(len(y_values))
    selector = projection @ feature
    if not np.isin(selector, (0, 1)).all() or int(selector.sum()) != 1:
        raise ValueError("hard-drop selector must be one-hot")

    live_hits = np.flatnonzero(selector[1:])
    y = -1 if selector[0] else y_values[int(live_hits[0])]
    return LinearizedHardDropResult(
        y_values=y_values,
        y=int(y),
        valid_vector=valid,
        landing_vector=landing,
        feature_state=feature,
        selector_projection=projection,
        selector=selector.astype(np.uint8, copy=False),
    )


def mask_cells(mask: np.ndarray | Iterable[int]) -> tuple[int, ...]:
    return tuple(int(idx) for idx in np.flatnonzero(as_board_vector(mask)))


def collision_occupancy_projection(cells: Iterable[int]) -> np.ndarray:
    cell_tuple = tuple(cells)
    pattern_count = 1 << len(cell_tuple)
    matrix = np.zeros((1 + pattern_count, 1 + pattern_count), dtype=np.int64)
    matrix[0, 0] = 1
    for pattern in range(pattern_count):
        for subset in range(pattern_count):
            if subset & pattern == pattern:
                sign = -1 if (subset ^ pattern).bit_count() % 2 else 1
                matrix[1 + pattern, 1 + subset] = sign
    return matrix


def collision_occupancy_lift(
    state: np.ndarray | Iterable[int], cells: Iterable[int]
) -> CollisionOccupancyLiftResult:
    x = as_state_vector(state)
    cell_tuple = tuple(cells)
    if any(cell < 0 or cell >= BOARD_SIZE for cell in cell_tuple):
        raise ValueError("collision cells must be board indices")

    board = board_from_state(x)
    monomials = np.zeros(1 << len(cell_tuple), dtype=np.int64)
    alive = int(x[ALIVE_INDEX])
    for subset in range(monomials.size):
        value = alive
        for idx, cell in enumerate(cell_tuple):
            if subset & (1 << idx):
                value *= int(board[cell])
        monomials[subset] = value

    lifted = np.concatenate([np.array([int(x[DEAD_INDEX])], dtype=np.int64), monomials])
    projection = collision_occupancy_projection(cell_tuple)
    tensor = projection @ lifted
    if not np.isin(tensor, (0, 1)).all() or int(tensor.sum()) != 1:
        raise ValueError("collision occupancy lift did not produce a one-hot tensor")

    return CollisionOccupancyLiftResult(
        cells=cell_tuple,
        lifted_state=lifted,
        projection_matrix=projection,
        local_occupancy_tensor=tensor.astype(np.uint8, copy=False),
    )


def local_occupancy_tensor(
    state: np.ndarray | Iterable[int], cells: Iterable[int]
) -> np.ndarray:
    return collision_occupancy_lift(state, cells).local_occupancy_tensor


def collision_selector_projection(cells: Iterable[int]) -> np.ndarray:
    cell_tuple = tuple(cells)
    projection = np.zeros((2, 1 + (1 << len(cell_tuple))), dtype=np.uint8)
    projection[0, 0] = 1
    projection[1, 1] = 1
    projection[0, 2:] = 1
    return projection


def linearized_collision(
    state: np.ndarray | Iterable[int], mask: np.ndarray | Iterable[int]
) -> LinearizedCollisionResult:
    cells = mask_cells(mask)
    occupancy_lift = collision_occupancy_lift(state, cells)
    tensor = occupancy_lift.local_occupancy_tensor
    projection = collision_selector_projection(cells)
    selector = projection @ tensor
    if int(selector.sum()) != 1:
        raise ValueError("collision selector must be one-hot")
    return LinearizedCollisionResult(
        cells=cells,
        selector=selector.astype(np.uint8, copy=False),
        local_occupancy_tensor=tensor,
        selector_projection=projection,
        occupancy_lift=occupancy_lift,
    )


def dead_collision_result() -> LinearizedCollisionResult:
    occupancy_lift = collision_occupancy_lift(make_dead_state(), ())
    projection = collision_selector_projection(())
    selector = projection @ occupancy_lift.local_occupancy_tensor
    return LinearizedCollisionResult(
        cells=(),
        selector=selector.astype(np.uint8, copy=False),
        local_occupancy_tensor=occupancy_lift.local_occupancy_tensor,
        selector_projection=projection,
        occupancy_lift=occupancy_lift,
    )


def row_count_shift_matrix(row_adds: np.ndarray | Iterable[int]) -> np.ndarray:
    adds = np.asarray(row_adds, dtype=np.int64)
    if adds.shape != (ROWS,):
        raise ValueError(f"row_adds must have shape ({ROWS},)")
    if np.any(adds < 0) or np.any(adds > COLS):
        raise ValueError("row_adds entries must be in 0..10")

    matrix = np.zeros((Q_SIZE, Q_SIZE), dtype=np.uint8)
    for row, add in enumerate(adds):
        out_base = row * COUNT_STATES
        in_base = row * COUNT_STATES
        for count in range(COUNT_STATES - int(add)):
            matrix[out_base + count + int(add), in_base + count] = 1
    return matrix


def lock_matrix(mask: np.ndarray | Iterable[int]) -> np.ndarray:
    mask_vec = as_board_vector(mask)
    row_adds = row_sum_matrix() @ mask_vec
    count_update = row_count_shift_matrix(row_adds)

    matrix = np.zeros((STATE_SIZE, STATE_SIZE), dtype=np.uint8)
    matrix[:BOARD_SIZE, :BOARD_SIZE] = np.eye(BOARD_SIZE, dtype=np.uint8)
    matrix[:BOARD_SIZE, HOMOG_INDEX] = mask_vec
    matrix[BOARD_SIZE : BOARD_SIZE + Q_SIZE, BOARD_SIZE : BOARD_SIZE + Q_SIZE] = count_update
    matrix[ALIVE_INDEX, ALIVE_INDEX] = 1
    matrix[DEAD_INDEX, DEAD_INDEX] = 1
    matrix[HOMOG_INDEX, HOMOG_INDEX] = 1
    return matrix


def identity_state_matrix() -> np.ndarray:
    return np.eye(STATE_SIZE, dtype=np.uint8)


def death_matrix() -> np.ndarray:
    matrix = np.zeros((STATE_SIZE, STATE_SIZE), dtype=np.uint8)
    matrix[:, HOMOG_INDEX] = make_dead_state()
    matrix[DEAD_INDEX, DEAD_INDEX] = 1
    return matrix


def normalize_monomial(monomial: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(int(value) for value in monomial)))
    if any(value < 0 or value >= STATE_SIZE for value in values):
        raise ValueError("monomial coordinate out of range")
    return values


def lift_state_to_monomials(
    state: np.ndarray | Iterable[int], basis: MonomialBasis
) -> np.ndarray:
    x = as_state_vector(state)
    z = np.zeros(basis.size, dtype=np.int64)
    for idx, monomial in enumerate(basis.monomials):
        value = 1
        for coord in monomial:
            value *= int(x[coord])
        z[idx] = value
    return z


def multiply_polynomials(
    left: dict[tuple[int, ...], int], right: dict[tuple[int, ...], int]
) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    for left_monomial, left_coeff in left.items():
        for right_monomial, right_coeff in right.items():
            monomial = normalize_monomial((*left_monomial, *right_monomial))
            coeff = left_coeff * right_coeff
            out[monomial] = out.get(monomial, 0) + coeff
            if out[monomial] == 0:
                del out[monomial]
    return out


def add_polynomials(
    *polynomials: dict[tuple[int, ...], int]
) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    for polynomial in polynomials:
        for monomial, coeff in polynomial.items():
            out[monomial] = out.get(monomial, 0) + coeff
            if out[monomial] == 0:
                del out[monomial]
    return out


def negate_polynomial(polynomial: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    return {monomial: -coeff for monomial, coeff in polynomial.items()}


def one_minus_polynomial(polynomial: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    return add_polynomials({(): 1}, negate_polynomial(polynomial))


def product_of_linear_factors(
    factors: Iterable[dict[tuple[int, ...], int]]
) -> dict[tuple[int, ...], int]:
    polynomial: dict[tuple[int, ...], int] = {(): 1}
    for factor in factors:
        polynomial = multiply_polynomials(polynomial, factor)
    return polynomial


def valid_collision_guard(cells: Iterable[int]) -> dict[tuple[int, ...], int]:
    factors: list[dict[tuple[int, ...], int]] = [{(ALIVE_INDEX,): 1}]
    for cell in cells:
        factors.append({(): 1, (int(cell),): -1})
    return product_of_linear_factors(factors)


def death_collision_guard(cells: Iterable[int]) -> dict[tuple[int, ...], int]:
    return one_minus_polynomial(valid_collision_guard(cells))


def linear_form_from_row(row: np.ndarray | Iterable[int]) -> dict[tuple[int, ...], int]:
    row_arr = np.asarray(row, dtype=np.int64)
    if row_arr.shape != (STATE_SIZE,):
        raise ValueError(f"row must have shape ({STATE_SIZE},)")

    form: dict[tuple[int, ...], int] = {}
    for coord in np.flatnonzero(row_arr):
        coeff = int(row_arr[coord])
        if coeff:
            form[(int(coord),)] = coeff
    return form


def monomial_pullback(
    linear_map: np.ndarray | Iterable[int], output_monomial: Iterable[int]
) -> dict[tuple[int, ...], int]:
    matrix = np.asarray(linear_map, dtype=np.int64)
    if matrix.shape != (STATE_SIZE, STATE_SIZE):
        raise ValueError(f"linear_map must have shape ({STATE_SIZE}, {STATE_SIZE})")

    polynomial: dict[tuple[int, ...], int] = {(): 1}
    for output_coord in normalize_monomial(output_monomial):
        form = linear_form_from_row(matrix[output_coord])
        polynomial = multiply_polynomials(polynomial, form)
    return polynomial


def monomial_transition_matrix(
    linear_map: np.ndarray | Iterable[int], basis: MonomialBasis
) -> SparseLinearMatrix:
    rows: list[dict[int, int]] = []
    for output_monomial in basis.monomials:
        polynomial = monomial_pullback(linear_map, output_monomial)
        row: dict[int, int] = {}
        for input_monomial, coeff in polynomial.items():
            input_idx = basis.index.get(input_monomial)
            if input_idx is None:
                raise ValueError(
                    "basis is not closed; missing monomial "
                    f"{input_monomial} needed for output {output_monomial}"
                )
            row[input_idx] = row.get(input_idx, 0) + coeff
        rows.append(row)
    return SparseLinearMatrix(shape=(basis.size, basis.size), rows=tuple(rows))


def persistent_z_transition_for_linear_map(
    state: np.ndarray | Iterable[int],
    linear_map: np.ndarray | Iterable[int],
    basis: MonomialBasis,
) -> PersistentZTransitionResult:
    z = lift_state_to_monomials(state, basis)
    transition = monomial_transition_matrix(linear_map, basis)
    z_next = transition @ z

    compact = np.zeros(STATE_SIZE, dtype=np.uint8)
    for coord in range(STATE_SIZE):
        idx = basis.index.get((coord,))
        if idx is None:
            raise ValueError("basis must include all singleton compact coordinates")
        compact[coord] = z_next[idx]

    return PersistentZTransitionResult(
        basis=basis,
        z_state=z,
        transition_matrix=transition,
        z_next=z_next,
        compact_next=compact,
    )


def placements_for_piece(piece: str) -> tuple[Placement, ...]:
    placements: list[Placement] = []
    for rotation, cells in enumerate(all_rotations(piece)):
        width = max(col for _, col in cells) + 1
        for column in range(COLS - width + 1):
            placements.append(Placement(piece, rotation, column))
    return tuple(placements)


def all_in_bounds_placements() -> tuple[Placement, ...]:
    placements: list[Placement] = []
    for piece in RUST_PIECE_ORDER:
        placements.extend(placements_for_piece(piece))
    return tuple(placements)


def evenly_spaced_subset(values: tuple[Placement, ...], count: int) -> tuple[Placement, ...]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count >= len(values):
        return values
    if count == 0:
        return ()
    return tuple(values[(idx * len(values)) // count] for idx in range(count))


def representative_action_placements(count: int = CANONICAL_ACTION_COUNT) -> tuple[Placement, ...]:
    grouped = {piece: placements_for_piece(piece) for piece in RUST_PIECE_ORDER}
    total = sum(len(group) for group in grouped.values())
    if count < 0 or count > total:
        raise ValueError(f"count must be in 0..{total}")

    floors: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for piece in RUST_PIECE_ORDER:
        numerator = count * len(grouped[piece])
        floors[piece] = numerator // total
        remainders.append((numerator % total, piece))

    remaining = count - sum(floors.values())
    for _, piece in sorted(remainders, reverse=True)[:remaining]:
        floors[piece] += 1

    placements: list[Placement] = []
    for piece in RUST_PIECE_ORDER:
        placements.extend(evenly_spaced_subset(grouped[piece], floors[piece]))
    if len(placements) != count:
        raise ValueError("representative action selection produced the wrong count")
    return tuple(placements)


def branch_descriptors_for_placement(placement: Placement) -> tuple[ActionMatrixBranch, ...]:
    try:
        y_values = tuple(
            placement_y_range(placement.piece, placement.rotation, placement.column)
        )
    except ValueError:
        return (
            ActionMatrixBranch(
                label=f"{placement}:dead",
                y=-1,
                clear_rows=(),
                is_death=True,
            ),
        )

    branches = [
        ActionMatrixBranch(
            label=f"{placement}:dead",
            y=-1,
            clear_rows=(),
            is_death=True,
        )
    ]
    for y in y_values:
        mask = piece_mask(placement.piece, placement.rotation, placement.column, y)
        for clear_rows in clear_patterns(touched_rows_from_mask(mask)):
            branches.append(
                ActionMatrixBranch(
                    label=f"{placement}:y={y}:clear={clear_rows}",
                    y=y,
                    clear_rows=clear_rows,
                )
            )
    return tuple(branches)


def action_matrix_for_placement(placement: Placement) -> BranchActionMatrix:
    return BranchActionMatrix(
        placement=placement,
        branches=branch_descriptors_for_placement(placement),
    )


def action_matrix_family(count: int = CANONICAL_ACTION_COUNT) -> ActionMatrixFamily:
    placements = representative_action_placements(count)
    matrices = tuple(action_matrix_for_placement(placement) for placement in placements)
    return ActionMatrixFamily(placements=placements, matrices=matrices)


def clear_pattern_guard_polynomial(
    lock: np.ndarray,
    candidate_rows: Iterable[int],
    clear_rows: Iterable[int],
) -> dict[tuple[int, ...], int]:
    clear_set = set(clear_rows)
    factors: list[dict[tuple[int, ...], int]] = []
    for row in tuple(sorted(set(candidate_rows))):
        full_coord = BOARD_SIZE + row * COUNT_STATES + COLS
        full_form = linear_form_from_row(lock[full_coord])
        factors.append(full_form if row in clear_set else one_minus_polynomial(full_form))
    return product_of_linear_factors(factors)


def branch_specs_for_placement(placement: Placement) -> tuple[BranchSpec, ...]:
    try:
        y_values = tuple(
            placement_y_range(placement.piece, placement.rotation, placement.column)
        )
    except ValueError:
        return (
            BranchSpec(
                label=f"{placement}:dead",
                guard={(): 1},
                matrix=death_matrix(),
            ),
        )

    specs = [
        BranchSpec(
            label=f"{placement}:dead",
            guard={(): 1},
            matrix=death_matrix(),
        )
    ]
    for y in y_values:
        mask = piece_mask(placement.piece, placement.rotation, placement.column, y)
        cells = mask_cells(mask)
        lock = lock_matrix(mask)
        rows = touched_rows_from_mask(mask)
        valid_guard = valid_collision_guard(cells)
        if y == 0:
            landing_guard = valid_guard
        else:
            lower_mask = piece_mask(
                placement.piece,
                placement.rotation,
                placement.column,
                y - 1,
            )
            landing_guard = multiply_polynomials(
                valid_guard,
                one_minus_polynomial(valid_collision_guard(mask_cells(lower_mask))),
            )
        for clear_rows in clear_patterns(rows):
            clear_guard = clear_pattern_guard_polynomial(lock, rows, clear_rows)
            specs.append(
                BranchSpec(
                    label=f"{placement}:y={y}:clear={clear_rows}",
                    guard=multiply_polynomials(landing_guard, clear_guard),
                    matrix=clear_matrix(clear_rows) @ lock,
                )
            )
    return tuple(specs)


def closure_degree_histogram(features: Iterable[tuple[int, ...]]) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for feature in features:
        degree = len(feature)
        histogram[degree] = histogram.get(degree, 0) + 1
    return dict(sorted(histogram.items()))


def explore_closed_monomial_basis(
    placements: Iterable[Placement],
    *,
    max_features: int,
    max_degree: int,
    max_processed: int,
) -> ClosureReport:
    placement_tuple = tuple(placements)
    branch_specs = tuple(
        branch for placement in placement_tuple for branch in branch_specs_for_placement(placement)
    )
    features: set[tuple[int, ...]] = {()}
    features.update((coord,) for coord in range(STATE_SIZE))
    queue: list[tuple[int, ...]] = list(features)
    processed = 0
    hit_feature_cap = False
    hit_degree_cap = False

    while queue and processed < max_processed and len(features) < max_features:
        feature = queue.pop(0)
        processed += 1
        for branch in branch_specs:
            pullback = monomial_pullback(branch.matrix, feature)
            guarded = multiply_polynomials(branch.guard, pullback)
            for monomial in guarded:
                if len(monomial) > max_degree:
                    hit_degree_cap = True
                    continue
                if monomial in features:
                    continue
                features.add(monomial)
                queue.append(monomial)
                if len(features) >= max_features:
                    hit_feature_cap = True
                    break
            if hit_feature_cap:
                break

    return ClosureReport(
        action_count=len(placement_tuple),
        branch_count=len(branch_specs),
        processed_features=processed,
        feature_count=len(features),
        max_degree=max((len(feature) for feature in features), default=0),
        hit_feature_cap=hit_feature_cap or len(features) >= max_features,
        hit_degree_cap=hit_degree_cap,
        degree_histogram=closure_degree_histogram(features),
    )


def full_rows_from_q(q: np.ndarray | Iterable[int]) -> tuple[int, ...]:
    blocks = np.asarray(q).reshape(ROWS, COUNT_STATES)
    return tuple(int(row) for row in np.flatnonzero(blocks[:, COLS]))


def touched_rows_from_mask(mask: np.ndarray | Iterable[int]) -> tuple[int, ...]:
    row_adds = row_sum_matrix() @ as_board_vector(mask)
    return tuple(int(row) for row in np.flatnonzero(row_adds))


def clear_patterns(candidate_rows: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(sorted(set(candidate_rows)))
    if any(row < 0 or row >= ROWS for row in rows):
        raise ValueError("candidate rows must be in 0..19")

    patterns: list[tuple[int, ...]] = []
    for bits in range(1 << len(rows)):
        patterns.append(tuple(row for idx, row in enumerate(rows) if bits & (1 << idx)))
    return tuple(patterns)


def clear_pattern_index(clear_rows: Iterable[int], candidate_rows: Iterable[int]) -> int:
    rows = tuple(sorted(set(candidate_rows)))
    clear_set = set(clear_rows)
    missing = clear_set - set(rows)
    if missing:
        raise ValueError(f"clear rows {sorted(missing)} are not in candidate rows {rows}")

    index = 0
    for idx, row in enumerate(rows):
        if row in clear_set:
            index |= 1 << idx
    return index


def count_tuple_index(counts: Iterable[int]) -> int:
    index = 0
    for count in counts:
        index = index * COUNT_STATES + int(count)
    return index


def local_count_tensor_projection(candidate_rows: Iterable[int]) -> np.ndarray:
    rows = tuple(sorted(set(candidate_rows)))
    size = COUNT_STATES ** len(rows)
    matrix = np.zeros((size, 1 + size), dtype=np.uint8)
    matrix[0, 0] = 1
    matrix[:, 1:] = np.eye(size, dtype=np.uint8)
    return matrix


def local_count_tensor_lift(
    state: np.ndarray | Iterable[int], candidate_rows: Iterable[int]
) -> LocalCountTensorLiftResult:
    x = as_state_vector(state)
    rows = tuple(sorted(set(candidate_rows)))
    if any(row < 0 or row >= ROWS for row in rows):
        raise ValueError("candidate rows must be in 0..19")

    q = q_from_state(x)
    size = COUNT_STATES ** len(rows)
    products = np.zeros(size, dtype=np.uint8)
    for counts in product(range(COUNT_STATES), repeat=len(rows)):
        value = int(x[ALIVE_INDEX])
        for row, count in zip(rows, counts):
            value *= int(q[row * COUNT_STATES + count])
        products[count_tuple_index(counts)] = value

    lifted = np.concatenate([np.array([int(x[DEAD_INDEX])], dtype=np.uint8), products])
    projection = local_count_tensor_projection(rows)
    tensor = projection @ lifted
    if not np.isin(tensor, (0, 1)).all() or int(tensor.sum()) != 1:
        raise ValueError("local count lift did not produce a one-hot tensor")

    return LocalCountTensorLiftResult(
        rows=rows,
        lifted_state=lifted,
        projection_matrix=projection,
        local_count_tensor=tensor.astype(np.uint8, copy=False),
    )


def local_count_tensor(q: np.ndarray | Iterable[int], candidate_rows: Iterable[int]) -> np.ndarray:
    rows = tuple(sorted(set(candidate_rows)))
    counts = counts_from_q_or_dead(q)
    size = COUNT_STATES ** len(rows)
    tensor = np.zeros(size, dtype=np.uint8)
    tensor[count_tuple_index(int(counts[row]) for row in rows)] = 1
    return tensor


def clear_selector_projection(candidate_rows: Iterable[int]) -> np.ndarray:
    rows = tuple(sorted(set(candidate_rows)))
    patterns = clear_patterns(rows)
    projection = np.zeros((len(patterns), COUNT_STATES ** len(rows)), dtype=np.uint8)

    for tensor_index, counts in enumerate(product(range(COUNT_STATES), repeat=len(rows))):
        clear_rows = tuple(row for row, count in zip(rows, counts) if count == COLS)
        projection[clear_pattern_index(clear_rows, rows), tensor_index] = 1
    return projection


def clear_selector_from_q(
    q: np.ndarray | Iterable[int], candidate_rows: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = local_count_tensor(q, candidate_rows)
    projection = clear_selector_projection(candidate_rows)
    selector = projection @ tensor
    return selector.astype(np.uint8, copy=False), tensor, projection


def clear_selector_from_state(
    state: np.ndarray | Iterable[int], candidate_rows: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LocalCountTensorLiftResult]:
    count_lift = local_count_tensor_lift(state, candidate_rows)
    projection = clear_selector_projection(candidate_rows)
    selector = projection @ count_lift.local_count_tensor
    return (
        selector.astype(np.uint8, copy=False),
        count_lift.local_count_tensor,
        projection,
        count_lift,
    )


def selector_product_from_quadratic_projection(left_size: int, right_size: int) -> np.ndarray:
    base_size = left_size + right_size
    matrix = np.zeros((left_size * right_size, base_size * base_size), dtype=np.uint8)
    for left_idx in range(left_size):
        for right_idx in range(right_size):
            out_idx = left_idx * right_size + right_idx
            quad_idx = left_idx * base_size + left_size + right_idx
            matrix[out_idx, quad_idx] = 1
    return matrix


def placement_selector_projection(clear_pattern_count: int) -> np.ndarray:
    matrix = np.zeros((1 + clear_pattern_count, 2 * clear_pattern_count), dtype=np.uint8)
    for clear_idx in range(clear_pattern_count):
        matrix[0, clear_idx] = 1
        matrix[1 + clear_idx, clear_pattern_count + clear_idx] = 1
    return matrix


def placement_selector_lift(
    collision_selector: np.ndarray | Iterable[int],
    clear_selector: np.ndarray | Iterable[int],
) -> PlacementSelectorLiftResult:
    collision = np.asarray(collision_selector, dtype=np.uint8)
    clear = np.asarray(clear_selector, dtype=np.uint8)
    if collision.shape != (2,):
        raise ValueError("collision selector must have shape (2,)")
    if clear.ndim != 1:
        raise ValueError("clear selector must be one-dimensional")
    if not np.isin(collision, (0, 1)).all() or int(collision.sum()) != 1:
        raise ValueError("collision selector must be one-hot")
    if not np.isin(clear, (0, 1)).all() or int(clear.sum()) != 1:
        raise ValueError("clear selector must be one-hot")

    base = np.concatenate([collision, clear]).astype(np.uint8, copy=False)
    quadratic = np.kron(base, base).astype(np.uint8, copy=False)
    product_projection = selector_product_from_quadratic_projection(2, clear.size)
    product_state = product_projection @ quadratic
    selector_projection = placement_selector_projection(clear.size)
    selector = selector_projection @ product_state
    if int(selector.sum()) != 1:
        raise ValueError("placement selector must be one-hot")

    return PlacementSelectorLiftResult(
        collision_selector=collision,
        clear_selector=clear,
        base_state=base,
        quadratic_state=quadratic,
        product_projection_matrix=product_projection,
        product_state=product_state.astype(np.uint8, copy=False),
        selector_projection_matrix=selector_projection,
        selector=selector.astype(np.uint8, copy=False),
    )


def guarded_copy_projection_matrix(selector_size: int) -> tuple[np.ndarray, int]:
    product_size = selector_size * STATE_SIZE
    product_offset = STATE_SIZE + selector_size
    matrix = np.zeros((product_size, product_offset + product_size), dtype=np.uint8)
    matrix[:, product_offset:] = np.eye(product_size, dtype=np.uint8)
    return matrix, product_offset


def guarded_product_from_quadratic_projection(selector_size: int) -> np.ndarray:
    base_size = STATE_SIZE + selector_size
    matrix = np.zeros((selector_size * STATE_SIZE, base_size * base_size), dtype=np.uint8)
    for selector_idx in range(selector_size):
        selector_base_idx = STATE_SIZE + selector_idx
        for state_idx in range(STATE_SIZE):
            out_idx = selector_idx * STATE_SIZE + state_idx
            quad_idx = selector_base_idx * base_size + state_idx
            matrix[out_idx, quad_idx] = 1
    return matrix


def quadratic_product_lift(
    state: np.ndarray | Iterable[int], selector: np.ndarray | Iterable[int]
) -> QuadraticProductLiftResult:
    x = as_state_vector(state)
    z = np.asarray(selector, dtype=np.uint8)
    if z.ndim != 1:
        raise ValueError("selector must be a one-dimensional vector")
    if not np.isin(z, (0, 1)).all() or int(z.sum()) != 1:
        raise ValueError("selector must be one-hot")

    base = np.concatenate([x, z]).astype(np.uint8, copy=False)
    quadratic = np.kron(base, base).astype(np.uint8, copy=False)
    projection = guarded_product_from_quadratic_projection(z.size)
    product = projection @ quadratic
    return QuadraticProductLiftResult(
        base_state=base,
        quadratic_state=quadratic,
        product_projection_matrix=projection,
        product_state=product.astype(np.uint8, copy=False),
    )


def guarded_copy_product_state(
    state: np.ndarray | Iterable[int], selector: np.ndarray | Iterable[int]
) -> np.ndarray:
    x = as_state_vector(state)
    z = np.asarray(selector, dtype=np.uint8)
    if z.ndim != 1:
        raise ValueError("selector must be a one-dimensional vector")
    if not np.isin(z, (0, 1)).all() or int(z.sum()) != 1:
        raise ValueError("selector must be one-hot")
    return np.kron(z, x).astype(np.uint8, copy=False)


def guarded_copy_lift_from_product(
    state: np.ndarray | Iterable[int],
    selector: np.ndarray | Iterable[int],
    product_state: np.ndarray | Iterable[int],
    *,
    quadratic_lift: QuadraticProductLiftResult | None = None,
) -> GuardedCopyLiftResult:
    x = as_state_vector(state)
    z = np.asarray(selector, dtype=np.uint8)
    product = np.asarray(product_state, dtype=np.uint8)
    if z.ndim != 1:
        raise ValueError("selector must be a one-dimensional vector")
    if not np.isin(z, (0, 1)).all() or int(z.sum()) != 1:
        raise ValueError("selector must be one-hot")
    if product.shape != (z.size * STATE_SIZE,):
        raise ValueError("product state has wrong shape")
    if not np.isin(product, (0, 1)).all():
        raise ValueError("product state must be binary")

    lifted = np.concatenate([x, z, product]).astype(np.uint8, copy=False)
    projection, product_offset = guarded_copy_projection_matrix(z.size)
    guarded = projection @ lifted
    return GuardedCopyLiftResult(
        selector=z,
        source_state=x,
        product_state=product,
        lifted_state=lifted,
        projection_matrix=projection,
        guarded_state=guarded.astype(np.uint8, copy=False),
        product_offset=product_offset,
        quadratic_lift=quadratic_lift,
    )


def guarded_copy_lift(
    state: np.ndarray | Iterable[int], selector: np.ndarray | Iterable[int]
) -> GuardedCopyLiftResult:
    quadratic = quadratic_product_lift(state, selector)
    return guarded_copy_lift_from_product(
        state,
        selector,
        quadratic.product_state,
        quadratic_lift=quadratic,
    )


def linearized_lock_result(
    source_state: np.ndarray | Iterable[int],
    lock: np.ndarray,
    collision_selector: np.ndarray | Iterable[int],
) -> LinearizedLockResult:
    x = as_state_vector(source_state)
    selector = np.asarray(collision_selector, dtype=np.uint8)
    if selector.shape != (2,):
        raise ValueError("collision selector must have shape (2,)")
    if not np.isin(selector, (0, 1)).all() or int(selector.sum()) != 1:
        raise ValueError("collision selector must be one-hot")
    if lock.shape != (STATE_SIZE, STATE_SIZE):
        raise ValueError("lock matrix has wrong shape")

    guarded_lift = guarded_copy_lift(x, selector)
    guarded = guarded_lift.guarded_state
    branch_matrix = np.concatenate([death_matrix(), lock], axis=1)
    next_state = branch_matrix @ guarded
    return LinearizedLockResult(
        next_state=next_state.astype(np.uint8, copy=False),
        selector=selector,
        guarded_state=guarded,
        guarded_lift=guarded_lift,
        branch_matrix=branch_matrix,
    )


def linearized_clear_matrix(clear_pattern_rows: Iterable[Iterable[int]]) -> np.ndarray:
    matrices = [clear_matrix(pattern) for pattern in clear_pattern_rows]
    if not matrices:
        raise ValueError("at least one clear pattern is required")
    return np.concatenate(matrices, axis=1)


def make_clear_lifted_state(
    locked_state: np.ndarray | Iterable[int],
    local_tensor: np.ndarray | Iterable[int],
    selector: np.ndarray | Iterable[int],
    guarded_state: np.ndarray | Iterable[int],
) -> tuple[np.ndarray, int]:
    x_lock = as_state_vector(locked_state)
    tensor = np.asarray(local_tensor, dtype=np.uint8)
    z = np.asarray(selector, dtype=np.uint8)
    guarded = np.asarray(guarded_state, dtype=np.uint8)

    guarded_offset = STATE_SIZE + tensor.size + z.size
    lifted = np.concatenate([x_lock, tensor, z, guarded]).astype(np.uint8, copy=False)
    return lifted, guarded_offset


def clear_state_matrix(
    clear_lift_matrix: np.ndarray,
    clear_lifted_state_size: int,
    guarded_state_offset: int,
) -> np.ndarray:
    matrix = np.zeros((STATE_SIZE, clear_lifted_state_size), dtype=np.uint8)
    guarded_width = clear_lift_matrix.shape[1]
    matrix[:, guarded_state_offset : guarded_state_offset + guarded_width] = clear_lift_matrix
    return matrix


def placement_state_matrix(
    branch_matrix: np.ndarray,
    lifted_state_size: int,
    guarded_state_offset: int,
) -> np.ndarray:
    matrix = np.zeros((STATE_SIZE, lifted_state_size), dtype=np.uint8)
    guarded_width = branch_matrix.shape[1]
    matrix[:, guarded_state_offset : guarded_state_offset + guarded_width] = branch_matrix
    return matrix


def linearized_step_lift_result(
    source_state: np.ndarray | Iterable[int],
    collision: LinearizedCollisionResult,
    linearized_lock: LinearizedLockResult,
    linearized_clear: LinearizedClearResult,
    placement_lift: PlacementLiftResult,
) -> LinearizedStepLiftResult:
    x = as_state_vector(source_state)
    selector_product = (
        placement_lift.selector_lift.product_state
        if placement_lift.selector_lift is not None
        else np.zeros(0, dtype=np.uint8)
    )

    components = (
        ("source_state", x),
        ("collision_monomial_lift", collision.occupancy_lift.lifted_state),
        ("collision_occupancy_tensor", collision.local_occupancy_tensor),
        ("collision_selector", collision.selector),
        ("lock_guarded_state", linearized_lock.guarded_state),
        ("locked_or_dead_state", linearized_lock.next_state),
        ("clear_count_lift", linearized_clear.count_lift.lifted_state),
        ("clear_count_tensor", linearized_clear.local_count_tensor),
        ("clear_selector", linearized_clear.selector),
        ("clear_guarded_state", linearized_clear.guarded_state),
        ("placement_selector_product", selector_product),
        ("placement_selector", placement_lift.selector),
        ("placement_guarded_state", placement_lift.guarded_state),
    )

    offsets: dict[str, int] = {}
    parts: list[np.ndarray] = []
    cursor = 0
    for name, component in components:
        vector = np.asarray(component, dtype=np.uint8).reshape(-1)
        offsets[name] = cursor
        parts.append(vector)
        cursor += vector.size

    lifted_state = np.concatenate(parts).astype(np.uint8, copy=False)
    guarded_offset = offsets["placement_guarded_state"]
    guarded_size = placement_lift.guarded_state.size
    step_matrix = np.zeros((STATE_SIZE, lifted_state.size), dtype=np.uint8)
    step_matrix[:, guarded_offset : guarded_offset + guarded_size] = placement_lift.branch_matrix
    next_state = step_matrix @ lifted_state

    return LinearizedStepLiftResult(
        next_state=next_state.astype(np.uint8, copy=False),
        lifted_state=lifted_state,
        step_matrix=step_matrix,
        component_offsets=offsets,
        guarded_state_offset=guarded_offset,
        guarded_state_size=guarded_size,
    )


def append_projection_block(
    blocks: list[ProjectionBlock],
    out_offset: int,
    in_offset: int,
    block: np.ndarray | Iterable[int],
) -> None:
    block_arr = np.asarray(block, dtype=np.int64)
    if block_arr.ndim != 2:
        raise ValueError("projection block must be two-dimensional")
    blocks.append(
        ProjectionBlock(
            out_offset=out_offset,
            in_offset=in_offset,
            block=block_arr,
        )
    )


def linearized_step_feature_result(
    source_state: np.ndarray | Iterable[int],
    collision: LinearizedCollisionResult,
    linearized_lock: LinearizedLockResult,
    linearized_clear: LinearizedClearResult,
    placement_lift: PlacementLiftResult,
    step_lift: LinearizedStepLiftResult,
) -> LinearizedStepFeatureResult:
    x = as_state_vector(source_state)
    if placement_lift.selector_lift is None:
        placement_selector_source = placement_lift.selector
    else:
        placement_selector_source = placement_lift.selector_lift.product_state

    components = (
        ("source_state", x),
        ("collision_monomial_lift", collision.occupancy_lift.lifted_state),
        ("lock_guarded_product", linearized_lock.guarded_lift.product_state),
        ("clear_count_lift", linearized_clear.count_lift.lifted_state),
        ("clear_guarded_product", linearized_clear.guarded_lift.product_state),
        ("placement_selector_source", placement_selector_source),
        ("placement_guarded_product", placement_lift.guarded_lift.product_state),
    )

    feature_offsets: dict[str, int] = {}
    feature_parts: list[np.ndarray] = []
    cursor = 0
    for name, component in components:
        vector = np.asarray(component, dtype=np.int64).reshape(-1)
        feature_offsets[name] = cursor
        feature_parts.append(vector)
        cursor += vector.size

    feature_state = np.concatenate(feature_parts).astype(np.int64, copy=False)
    projection_blocks: list[ProjectionBlock] = []

    source_offset = feature_offsets["source_state"]
    collision_offset = feature_offsets["collision_monomial_lift"]
    lock_product_offset = feature_offsets["lock_guarded_product"]
    clear_count_offset = feature_offsets["clear_count_lift"]
    clear_product_offset = feature_offsets["clear_guarded_product"]
    placement_selector_offset = feature_offsets["placement_selector_source"]
    placement_product_offset = feature_offsets["placement_guarded_product"]

    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["source_state"],
        source_offset,
        np.eye(STATE_SIZE, dtype=np.int64),
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["collision_monomial_lift"],
        collision_offset,
        np.eye(collision.occupancy_lift.lifted_state.size, dtype=np.int64),
    )

    collision_tensor_matrix = collision.occupancy_lift.projection_matrix
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["collision_occupancy_tensor"],
        collision_offset,
        collision_tensor_matrix,
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["collision_selector"],
        collision_offset,
        collision.selector_projection.astype(np.int64) @ collision_tensor_matrix,
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["lock_guarded_state"],
        lock_product_offset,
        np.eye(linearized_lock.guarded_state.size, dtype=np.int64),
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["locked_or_dead_state"],
        lock_product_offset,
        linearized_lock.branch_matrix,
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["clear_count_lift"],
        clear_count_offset,
        np.eye(linearized_clear.count_lift.lifted_state.size, dtype=np.int64),
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["clear_count_tensor"],
        clear_count_offset,
        linearized_clear.count_lift.projection_matrix,
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["clear_selector"],
        clear_count_offset,
        linearized_clear.selector_projection.astype(np.int64)
        @ linearized_clear.count_lift.projection_matrix,
    )
    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["clear_guarded_state"],
        clear_product_offset,
        np.eye(linearized_clear.guarded_state.size, dtype=np.int64),
    )

    placement_selector_product_size = step_lift.component_offsets["placement_selector"]
    placement_selector_product_size -= step_lift.component_offsets["placement_selector_product"]
    if placement_lift.selector_lift is None:
        append_projection_block(
            projection_blocks,
            step_lift.component_offsets["placement_selector"],
            placement_selector_offset,
            np.eye(placement_lift.selector.size, dtype=np.int64),
        )
    else:
        append_projection_block(
            projection_blocks,
            step_lift.component_offsets["placement_selector_product"],
            placement_selector_offset,
            np.eye(placement_selector_product_size, dtype=np.int64),
        )
        append_projection_block(
            projection_blocks,
            step_lift.component_offsets["placement_selector"],
            placement_selector_offset,
            placement_lift.selector_lift.selector_projection_matrix,
        )

    append_projection_block(
        projection_blocks,
        step_lift.component_offsets["placement_guarded_state"],
        placement_product_offset,
        np.eye(placement_lift.guarded_state.size, dtype=np.int64),
    )

    projection = BlockProjectionMatrix(
        shape=(step_lift.lifted_state.size, feature_state.size),
        blocks=tuple(projection_blocks),
    )
    projected_step_lift = projection @ feature_state
    feature_step_matrix = ComposedFeatureStepMatrix(
        left=step_lift.step_matrix,
        right=projection,
    )
    next_state = feature_step_matrix @ feature_state

    return LinearizedStepFeatureResult(
        next_state=next_state.astype(np.uint8, copy=False),
        feature_state=feature_state,
        feature_to_step_matrix=projection,
        projected_step_lift=projected_step_lift,
        feature_step_matrix=feature_step_matrix,
        feature_offsets=feature_offsets,
    )


def linearized_placement_result(
    source_state: np.ndarray | Iterable[int],
    lock: np.ndarray,
    clear_pattern_rows: Iterable[Iterable[int]],
    clear_selector: np.ndarray | Iterable[int],
    *,
    force_dead: bool = False,
    collision_selector: np.ndarray | Iterable[int] | None = None,
    guarded_product: np.ndarray | Iterable[int] | None = None,
) -> PlacementLiftResult:
    x = as_state_vector(source_state)
    patterns = tuple(tuple(pattern) for pattern in clear_pattern_rows)
    selector_tail = np.asarray(clear_selector, dtype=np.uint8)
    if selector_tail.shape != (len(patterns),):
        raise ValueError("clear selector length must match clear pattern count")

    if collision_selector is None:
        collision_selector = np.array([1, 0] if force_dead else [0, 1], dtype=np.uint8)
    selector_lift = placement_selector_lift(collision_selector, selector_tail)
    selector = selector_lift.selector

    live_matrices = [clear_matrix(pattern) @ lock for pattern in patterns]
    branch_matrix = np.concatenate([death_matrix(), *live_matrices], axis=1)
    guarded_lift = (
        guarded_copy_lift(x, selector)
        if guarded_product is None
        else guarded_copy_lift_from_product(x, selector, guarded_product)
    )
    guarded = guarded_lift.guarded_state
    lifted_state = guarded_lift.lifted_state
    guarded_offset = guarded_lift.product_offset
    lifted_matrix = placement_state_matrix(branch_matrix, lifted_state.size, guarded_offset)
    next_state = lifted_matrix @ lifted_state
    labels = ("dead",) + tuple(f"clear={pattern}" for pattern in patterns)
    return PlacementLiftResult(
        next_state=next_state.astype(np.uint8, copy=False),
        selector=selector,
        branch_labels=labels,
        guarded_state=guarded,
        guarded_lift=guarded_lift,
        branch_matrix=branch_matrix,
        lifted_state=lifted_state,
        state_matrix=lifted_matrix,
        guarded_state_offset=guarded_offset,
        selector_lift=selector_lift,
    )


def dead_placement_lift_result(
    source_state: np.ndarray | Iterable[int],
    guarded_product: np.ndarray | Iterable[int] | None = None,
) -> PlacementLiftResult:
    x = as_state_vector(source_state)
    selector = np.array([1], dtype=np.uint8)
    guarded_lift = (
        guarded_copy_lift(x, selector)
        if guarded_product is None
        else guarded_copy_lift_from_product(x, selector, guarded_product)
    )
    guarded = guarded_lift.guarded_state
    branch_matrix = death_matrix()
    lifted_state = guarded_lift.lifted_state
    guarded_offset = guarded_lift.product_offset
    lifted_matrix = placement_state_matrix(branch_matrix, lifted_state.size, guarded_offset)
    next_state = lifted_matrix @ lifted_state
    return PlacementLiftResult(
        next_state=next_state.astype(np.uint8, copy=False),
        selector=selector,
        branch_labels=("dead",),
        guarded_state=guarded,
        guarded_lift=guarded_lift,
        branch_matrix=branch_matrix,
        lifted_state=lifted_state,
        state_matrix=lifted_matrix,
        guarded_state_offset=guarded_offset,
    )


def linearized_line_clear(
    locked_state: np.ndarray | Iterable[int],
    candidate_rows: Iterable[int],
    *,
    guarded_product: np.ndarray | Iterable[int] | None = None,
) -> LinearizedClearResult:
    x_lock = as_state_vector(locked_state)
    rows = tuple(sorted(set(candidate_rows)))
    actual_clear_rows = set(full_rows_from_q(q_from_state(x_lock)))
    missing_rows = actual_clear_rows - set(rows)
    if missing_rows:
        raise ValueError(
            "locked state has full rows outside the local clear lift: "
            f"{sorted(missing_rows)}. Include those rows or require the usual "
            "post-clear invariant before locking."
        )

    patterns = clear_patterns(rows)
    selector, tensor, projection, count_lift = clear_selector_from_state(x_lock, rows)
    guarded_lift = (
        guarded_copy_lift(x_lock, selector)
        if guarded_product is None
        else guarded_copy_lift_from_product(x_lock, selector, guarded_product)
    )
    guarded = guarded_lift.guarded_state
    matrix = linearized_clear_matrix(patterns)
    lifted_state, guarded_offset = make_clear_lifted_state(x_lock, tensor, selector, guarded)
    lifted_matrix = clear_state_matrix(matrix, lifted_state.size, guarded_offset)
    next_state = lifted_matrix @ lifted_state
    return LinearizedClearResult(
        next_state=next_state.astype(np.uint8, copy=False),
        candidate_rows=rows,
        clear_patterns=patterns,
        selector=selector,
        local_count_tensor=tensor,
        count_lift=count_lift,
        selector_projection=projection,
        guarded_state=guarded,
        guarded_lift=guarded_lift,
        clear_lift_matrix=matrix,
        clear_lifted_state=lifted_state,
        clear_state_matrix=lifted_matrix,
        guarded_state_offset=guarded_offset,
    )


def board_compaction_matrix(clear_rows: Iterable[int]) -> np.ndarray:
    clear_set = set(clear_rows)
    if any(row < 0 or row >= ROWS for row in clear_set):
        raise ValueError("clear rows must be in 0..19")

    survivors = [row for row in range(ROWS) if row not in clear_set]
    matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    for new_row, old_row in enumerate(survivors):
        for col in range(COLS):
            matrix[cell_index(new_row, col), cell_index(old_row, col)] = 1
    return matrix


def q_compaction_matrix(clear_rows: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    clear_set = set(clear_rows)
    if any(row < 0 or row >= ROWS for row in clear_set):
        raise ValueError("clear rows must be in 0..19")

    survivors = [row for row in range(ROWS) if row not in clear_set]
    matrix = np.zeros((Q_SIZE, Q_SIZE), dtype=np.uint8)
    eta = np.zeros(Q_SIZE, dtype=np.uint8)

    for new_row, old_row in enumerate(survivors):
        out_base = new_row * COUNT_STATES
        in_base = old_row * COUNT_STATES
        matrix[out_base : out_base + COUNT_STATES, in_base : in_base + COUNT_STATES] = np.eye(
            COUNT_STATES, dtype=np.uint8
        )

    for new_row in range(len(survivors), ROWS):
        eta[new_row * COUNT_STATES] = 1

    return matrix, eta


def clear_matrix(clear_rows: Iterable[int]) -> np.ndarray:
    rows = tuple(sorted(set(clear_rows)))
    board_shift = board_compaction_matrix(rows)
    q_shift, eta = q_compaction_matrix(rows)

    matrix = np.zeros((STATE_SIZE, STATE_SIZE), dtype=np.uint8)
    matrix[:BOARD_SIZE, :BOARD_SIZE] = board_shift
    matrix[BOARD_SIZE : BOARD_SIZE + Q_SIZE, BOARD_SIZE : BOARD_SIZE + Q_SIZE] = q_shift
    matrix[BOARD_SIZE : BOARD_SIZE + Q_SIZE, HOMOG_INDEX] = eta
    matrix[ALIVE_INDEX, ALIVE_INDEX] = 1
    matrix[DEAD_INDEX, DEAD_INDEX] = 1
    matrix[HOMOG_INDEX, HOMOG_INDEX] = 1
    return matrix


def dead_linearized_result(
    source_state: np.ndarray | Iterable[int],
    guarded_product: np.ndarray | Iterable[int] | None = None,
) -> LinearizedClearResult:
    x = as_state_vector(source_state)
    matrix = death_matrix()
    count_lift = local_count_tensor_lift(x, ())
    tensor = count_lift.local_count_tensor
    selector = np.array([1], dtype=np.uint8)
    guarded_lift = (
        guarded_copy_lift(x, selector)
        if guarded_product is None
        else guarded_copy_lift_from_product(x, selector, guarded_product)
    )
    guarded = guarded_lift.guarded_state
    lifted_state, guarded_offset = make_clear_lifted_state(x, tensor, selector, guarded)
    lifted_matrix = clear_state_matrix(matrix, lifted_state.size, guarded_offset)
    next_state = lifted_matrix @ lifted_state
    return LinearizedClearResult(
        next_state=next_state.astype(np.uint8, copy=False),
        candidate_rows=(),
        clear_patterns=((),),
        selector=selector,
        local_count_tensor=tensor,
        count_lift=count_lift,
        selector_projection=np.array([[1]], dtype=np.uint8),
        guarded_state=guarded,
        guarded_lift=guarded_lift,
        clear_lift_matrix=matrix,
        clear_lifted_state=lifted_state,
        clear_state_matrix=lifted_matrix,
        guarded_state_offset=guarded_offset,
    )


def dead_step_result(
    source_state: np.ndarray | Iterable[int],
    y: int,
    hard_drop: LinearizedHardDropResult | None = None,
) -> StepResult:
    x = as_state_vector(source_state)
    hard_drop_result = hard_drop if hard_drop is not None else dead_hard_drop_result()
    collision = dead_collision_result()
    linearized_lock = linearized_lock_result(x, death_matrix(), collision.selector)
    linearized = dead_linearized_result(source_state)
    placement_lift = dead_placement_lift_result(source_state)
    step_lift = linearized_step_lift_result(
        x,
        collision,
        linearized_lock,
        linearized,
        placement_lift,
    )
    step_feature = linearized_step_feature_result(
        x,
        collision,
        linearized_lock,
        linearized,
        placement_lift,
        step_lift,
    )
    return StepResult(
        next_state=step_feature.next_state,
        y=y,
        clear_rows=(),
        lock_matrix=death_matrix(),
        clear_matrix=identity_state_matrix(),
        linearized_lock=linearized_lock,
        linearized_collision=collision,
        linearized_clear=linearized,
        placement_lift=placement_lift,
        step_lift=step_lift,
        step_feature=step_feature,
        linearized_hard_drop=hard_drop_result,
        died=True,
    )


def direct_lock_and_clear(
    board: np.ndarray | Iterable[int], mask: np.ndarray | Iterable[int]
) -> tuple[np.ndarray, tuple[int, ...]]:
    board_vec = as_board_vector(board)
    mask_vec = as_board_vector(mask)
    if not placement_valid(board_vec, mask_vec):
        raise ValueError("piece mask collides with the board")

    locked = (board_vec + mask_vec).reshape(ROWS, COLS)
    clear_rows = tuple(int(row) for row in np.flatnonzero(locked.sum(axis=1) == COLS))
    survivors = [row for row in range(ROWS) if row not in set(clear_rows)]

    next_grid = np.zeros((ROWS, COLS), dtype=np.uint8)
    for new_row, old_row in enumerate(survivors):
        next_grid[new_row] = locked[old_row]
    return next_grid.reshape(BOARD_SIZE), clear_rows


def tetris_step(state: np.ndarray | Iterable[int], placement: Placement) -> StepResult:
    x = as_state_vector(state)
    hard_drop = linearized_hard_drop(x, placement)
    if hard_drop.selector[0]:
        return dead_step_result(x, hard_drop.y, hard_drop)

    y = hard_drop.y
    mask = piece_mask(placement.piece, placement.rotation, placement.column, y)

    collision = linearized_collision(x, mask)
    raw_lock_matrix = lock_matrix(mask)
    linearized_lock = linearized_lock_result(x, raw_lock_matrix, collision.selector)
    locked_or_dead = linearized_lock.next_state
    clear_rows = full_rows_from_q(q_from_state(locked_or_dead))
    k_matrix = clear_matrix(clear_rows)
    linearized_clear = linearized_line_clear(locked_or_dead, touched_rows_from_mask(mask))
    placement_lift = linearized_placement_result(
        x,
        raw_lock_matrix,
        linearized_clear.clear_patterns,
        linearized_clear.selector,
        collision_selector=collision.selector,
    )
    step_lift = linearized_step_lift_result(
        x,
        collision,
        linearized_lock,
        linearized_clear,
        placement_lift,
    )
    step_feature = linearized_step_feature_result(
        x,
        collision,
        linearized_lock,
        linearized_clear,
        placement_lift,
        step_lift,
    )
    next_state = step_feature.next_state
    effective_lock_matrix = (
        int(collision.selector[0]) * death_matrix()
        + int(collision.selector[1]) * raw_lock_matrix
    ).astype(np.uint8)
    return StepResult(
        next_state=next_state.astype(np.uint8, copy=False),
        y=y,
        clear_rows=clear_rows,
        lock_matrix=effective_lock_matrix,
        clear_matrix=k_matrix,
        linearized_lock=linearized_lock,
        linearized_collision=collision,
        linearized_clear=linearized_clear,
        placement_lift=placement_lift,
        step_lift=step_lift,
        step_feature=step_feature,
        linearized_hard_drop=hard_drop,
        died=is_dead_state(next_state),
    )


def selected_action_branch_index(
    state: np.ndarray | Iterable[int],
    action_matrix: BranchActionMatrix,
) -> int:
    x = as_state_vector(state)
    if is_dead_state(x):
        return 0

    hard_drop = linearized_hard_drop(x, action_matrix.placement)
    if hard_drop.selector[0]:
        return 0

    mask = piece_mask(
        action_matrix.placement.piece,
        action_matrix.placement.rotation,
        action_matrix.placement.column,
        hard_drop.y,
    )
    locked = lock_matrix(mask) @ x
    clear_rows = full_rows_from_q(q_from_state(locked))
    for branch_idx, branch in enumerate(action_matrix.branches):
        if branch.y == hard_drop.y and branch.clear_rows == clear_rows:
            return branch_idx
    raise ValueError("no action branch matches the state transition")


def action_branch_lifted_state(
    state: np.ndarray | Iterable[int],
    action_matrix: BranchActionMatrix,
) -> np.ndarray:
    x = as_state_vector(state)
    branch_idx = selected_action_branch_index(x, action_matrix)
    lifted = np.zeros(action_matrix.shape[1], dtype=np.uint8)
    start = branch_idx * STATE_SIZE
    lifted[start : start + STATE_SIZE] = x
    return lifted


def render_bottom(board: np.ndarray | Iterable[int], rows: int = 6) -> str:
    grid = board_grid(board)
    shown = []
    for row in range(min(rows, ROWS) - 1, -1, -1):
        shown.append("".join("#" if value else "." for value in grid[row]))
    return "\n".join(shown)


def assert_raises_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def assert_collision_lift_consistent(collision: LinearizedCollisionResult) -> None:
    expected_width = 1 + (1 << len(collision.cells))
    occupancy_lift = collision.occupancy_lift
    assert occupancy_lift.cells == collision.cells
    assert occupancy_lift.lifted_state.shape == (expected_width,)
    assert occupancy_lift.projection_matrix.shape == (expected_width, expected_width)
    assert np.array_equal(
        occupancy_lift.local_occupancy_tensor,
        occupancy_lift.projection_matrix @ occupancy_lift.lifted_state,
    )
    assert np.array_equal(collision.local_occupancy_tensor, occupancy_lift.local_occupancy_tensor)
    assert collision.local_occupancy_tensor.shape == (expected_width,)
    assert collision.selector_projection.shape == (2, expected_width)
    assert np.isin(collision.local_occupancy_tensor, (0, 1)).all()
    assert int(collision.local_occupancy_tensor.sum()) == 1
    assert np.array_equal(
        collision.selector,
        collision.selector_projection @ collision.local_occupancy_tensor,
    )
    assert collision.selector.shape == (2,)
    assert int(collision.selector.sum()) == 1


def assert_hard_drop_lift_consistent(hard_drop: LinearizedHardDropResult) -> None:
    y_count = len(hard_drop.y_values)
    assert hard_drop.valid_vector.shape == (y_count,)
    assert hard_drop.landing_vector.shape == (y_count,)
    assert hard_drop.feature_state.shape == (1 + 2 * y_count,)
    assert hard_drop.selector_projection.shape == (1 + y_count, 1 + 2 * y_count)
    assert hard_drop.selector.shape == (1 + y_count,)
    assert np.isin(hard_drop.valid_vector, (0, 1)).all()
    assert np.isin(hard_drop.landing_vector, (0, 1)).all()
    assert np.isin(hard_drop.selector, (0, 1)).all()
    assert int(hard_drop.feature_state[0]) == 1
    assert np.array_equal(
        hard_drop.feature_state[1 : 1 + y_count],
        hard_drop.valid_vector,
    )
    assert np.array_equal(
        hard_drop.feature_state[1 + y_count :],
        hard_drop.landing_vector,
    )
    assert np.array_equal(
        hard_drop.landing_vector,
        landing_vector_from_valid(hard_drop.valid_vector),
    )
    assert np.array_equal(
        hard_drop.selector,
        hard_drop.selector_projection @ hard_drop.feature_state,
    )
    assert int(hard_drop.selector.sum()) == 1
    if hard_drop.selector[0]:
        assert hard_drop.y == -1
    else:
        live_idx = int(np.flatnonzero(hard_drop.selector[1:])[0])
        assert hard_drop.y == hard_drop.y_values[live_idx]


def assert_placement_selector_lift_consistent(lift: PlacementSelectorLiftResult) -> None:
    clear_size = lift.clear_selector.size
    base_size = 2 + clear_size
    assert lift.collision_selector.shape == (2,)
    assert lift.base_state.shape == (base_size,)
    assert lift.quadratic_state.shape == (base_size * base_size,)
    assert lift.product_projection_matrix.shape == (2 * clear_size, base_size * base_size)
    assert lift.product_state.shape == (2 * clear_size,)
    assert lift.selector_projection_matrix.shape == (1 + clear_size, 2 * clear_size)
    assert lift.selector.shape == (1 + clear_size,)
    assert np.array_equal(lift.product_state, lift.product_projection_matrix @ lift.quadratic_state)
    assert np.array_equal(lift.product_state, np.kron(lift.collision_selector, lift.clear_selector))
    assert np.array_equal(lift.selector, lift.selector_projection_matrix @ lift.product_state)
    assert int(lift.selector.sum()) == 1


def assert_linearized_lock_consistent(lock: LinearizedLockResult) -> None:
    assert lock.selector.shape == (2,)
    assert int(lock.selector.sum()) == 1
    assert lock.branch_matrix.shape == (STATE_SIZE, 2 * STATE_SIZE)
    assert lock.guarded_state.shape == (2 * STATE_SIZE,)
    assert_guarded_lift_consistent(lock.guarded_lift)
    assert np.array_equal(lock.guarded_state, lock.guarded_lift.guarded_state)
    assert np.array_equal(lock.next_state, lock.branch_matrix @ lock.guarded_state)
    assert np.array_equal(lock.selector, lock.guarded_lift.selector)


def assert_local_count_lift_consistent(lift: LocalCountTensorLiftResult) -> None:
    expected_size = COUNT_STATES ** len(lift.rows)
    assert lift.lifted_state.shape == (1 + expected_size,)
    assert lift.projection_matrix.shape == (expected_size, 1 + expected_size)
    assert lift.local_count_tensor.shape == (expected_size,)
    assert np.isin(lift.lifted_state, (0, 1)).all()
    assert np.isin(lift.local_count_tensor, (0, 1)).all()
    assert np.array_equal(lift.local_count_tensor, lift.projection_matrix @ lift.lifted_state)
    assert int(lift.local_count_tensor.sum()) == 1


def assert_step_lift_consistent(step_lift: LinearizedStepLiftResult) -> None:
    assert step_lift.step_matrix.shape == (STATE_SIZE, step_lift.lifted_state.size)
    assert step_lift.guarded_state_size > 0
    assert np.array_equal(step_lift.next_state, step_lift.step_matrix @ step_lift.lifted_state)
    assert "placement_guarded_state" in step_lift.component_offsets
    assert (
        step_lift.component_offsets["placement_guarded_state"]
        == step_lift.guarded_state_offset
    )
    assert (
        step_lift.guarded_state_offset + step_lift.guarded_state_size
        <= step_lift.lifted_state.size
    )


def assert_step_result_lift_consistent(result: StepResult) -> None:
    assert_hard_drop_lift_consistent(result.linearized_hard_drop)
    assert result.y == result.linearized_hard_drop.y
    assert_step_lift_consistent(result.step_lift)
    start = result.step_lift.guarded_state_offset
    end = start + result.step_lift.guarded_state_size
    assert np.array_equal(
        result.step_lift.lifted_state[start:end],
        result.placement_lift.guarded_state,
    )
    assert np.array_equal(result.next_state, result.step_lift.next_state)
    assert np.array_equal(
        result.next_state,
        result.step_lift.step_matrix @ result.step_lift.lifted_state,
    )
    assert result.step_feature.feature_to_step_matrix.shape == (
        result.step_lift.lifted_state.size,
        result.step_feature.feature_state.size,
    )
    assert result.step_feature.feature_step_matrix.shape == (
        STATE_SIZE,
        result.step_feature.feature_state.size,
    )
    assert np.array_equal(
        result.step_lift.lifted_state,
        result.step_feature.projected_step_lift,
    )
    assert np.array_equal(
        result.step_lift.lifted_state,
        result.step_feature.feature_to_step_matrix @ result.step_feature.feature_state,
    )
    assert np.array_equal(
        result.next_state,
        result.step_feature.next_state,
    )
    assert np.array_equal(
        result.next_state,
        result.step_feature.feature_step_matrix @ result.step_feature.feature_state,
    )


def assert_action_matrix_matches_step(
    state: np.ndarray | Iterable[int],
    placement: Placement,
) -> None:
    x = as_state_vector(state)
    matrix = action_matrix_for_placement(placement)
    z = action_branch_lifted_state(x, matrix)
    result = tetris_step(x, placement)
    branch_idx = selected_action_branch_index(x, matrix)

    assert matrix.shape == (STATE_SIZE, len(matrix.branches) * STATE_SIZE)
    assert z.shape == (matrix.shape[1],)
    assert np.array_equal(z[branch_idx * STATE_SIZE : (branch_idx + 1) * STATE_SIZE], x)
    assert int(np.count_nonzero(z)) == int(np.count_nonzero(x))
    assert np.array_equal(matrix @ z, result.next_state)


def assert_persistent_degree_one_z_consistent(
    state: np.ndarray | Iterable[int], result: StepResult
) -> None:
    basis = MonomialBasis.from_coordinates(range(STATE_SIZE), max_degree=1)
    persistent = persistent_z_transition_for_linear_map(state, result.branch_matrix, basis)
    assert persistent.basis.size == STATE_SIZE + 1
    assert persistent.transition_matrix.shape == (STATE_SIZE + 1, STATE_SIZE + 1)
    assert persistent.z_state[basis.index[()]] == 1
    assert persistent.z_next[basis.index[()]] == 1
    assert np.array_equal(persistent.compact_next, result.next_state)
    assert np.array_equal(persistent.z_next, persistent.transition_matrix @ persistent.z_state)


def expected_clear_lifted_size(candidate_row_count: int) -> int:
    tensor_size = COUNT_STATES**candidate_row_count
    selector_size = 1 << candidate_row_count
    guarded_size = selector_size * STATE_SIZE
    return STATE_SIZE + tensor_size + selector_size + guarded_size


def expected_placement_lifted_size(branch_count: int) -> int:
    return STATE_SIZE + branch_count + branch_count * STATE_SIZE


def assert_guarded_lift_consistent(lift: GuardedCopyLiftResult) -> None:
    branch_count = lift.selector.size
    expected_size = STATE_SIZE + branch_count + branch_count * STATE_SIZE
    assert lift.product_state.shape == (branch_count * STATE_SIZE,)
    assert lift.lifted_state.shape == (expected_size,)
    assert lift.projection_matrix.shape == (branch_count * STATE_SIZE, expected_size)
    assert lift.product_offset == STATE_SIZE + branch_count
    assert np.array_equal(lift.product_state, lift.lifted_state[lift.product_offset :])
    assert np.array_equal(lift.guarded_state, lift.projection_matrix @ lift.lifted_state)
    assert np.array_equal(lift.product_state, np.kron(lift.selector, lift.source_state))
    assert np.array_equal(lift.guarded_state, lift.product_state)
    if lift.quadratic_lift is not None:
        quadratic = lift.quadratic_lift
        base_size = STATE_SIZE + branch_count
        assert quadratic.base_state.shape == (base_size,)
        assert quadratic.quadratic_state.shape == (base_size * base_size,)
        assert quadratic.product_projection_matrix.shape == (
            branch_count * STATE_SIZE,
            base_size * base_size,
        )
        assert np.array_equal(
            quadratic.product_state,
            quadratic.product_projection_matrix @ quadratic.quadratic_state,
        )
        assert np.array_equal(lift.product_state, quadratic.product_state)


def assert_live_step_matches_direct(
    board: np.ndarray, placement: Placement, expected_clear_rows: tuple[int, ...]
) -> StepResult:
    state = make_state(board)
    result = tetris_step(state, placement)
    mask = piece_mask(placement.piece, placement.rotation, placement.column, result.y)
    direct_board, direct_clear_rows = direct_lock_and_clear(board, mask)
    candidate_count = len(touched_rows_from_mask(mask))

    assert result.clear_rows == direct_clear_rows == expected_clear_rows
    assert result.died is False
    assert is_alive_state(result.next_state)
    assert result.linearized_hard_drop.selector[0] == 0
    assert_collision_lift_consistent(result.linearized_collision)
    assert result.linearized_collision.selector.tolist() == [0, 1]
    assert result.linearized_collision.local_occupancy_tensor[1] == 1
    assert_linearized_lock_consistent(result.linearized_lock)
    assert np.array_equal(result.linearized_lock.next_state, result.lock_matrix @ state)
    assert np.array_equal(board_from_state(result.next_state), direct_board)
    assert np.array_equal(result.next_state, result.branch_matrix @ state)
    assert np.array_equal(result.next_state, result.linearized_clear.next_state)
    assert_step_result_lift_consistent(result)
    assert_persistent_degree_one_z_consistent(state, result)
    locked = result.lock_matrix @ state
    clear_from_product = linearized_line_clear(
        locked,
        touched_rows_from_mask(mask),
        guarded_product=result.linearized_clear.guarded_lift.product_state,
    )
    assert np.array_equal(result.next_state, clear_from_product.next_state)
    assert np.array_equal(
        result.next_state,
        result.linearized_clear.clear_lift_matrix @ result.linearized_clear.guarded_state,
    )
    assert_local_count_lift_consistent(result.linearized_clear.count_lift)
    assert result.linearized_clear.count_lift.rows == result.linearized_clear.candidate_rows
    assert np.array_equal(
        result.linearized_clear.selector,
        result.linearized_clear.selector_projection @ result.linearized_clear.local_count_tensor,
    )
    assert_guarded_lift_consistent(result.linearized_clear.guarded_lift)
    assert np.array_equal(
        result.linearized_clear.guarded_state,
        result.linearized_clear.guarded_lift.guarded_state,
    )
    assert np.array_equal(
        result.next_state,
        result.linearized_clear.clear_state_matrix @ result.linearized_clear.clear_lifted_state,
    )
    assert np.array_equal(
        result.next_state,
        result.placement_lift.branch_matrix @ result.placement_lift.guarded_state,
    )
    assert_guarded_lift_consistent(result.placement_lift.guarded_lift)
    assert result.placement_lift.selector_lift is not None
    assert_placement_selector_lift_consistent(result.placement_lift.selector_lift)
    assert np.array_equal(
        result.placement_lift.selector_lift.collision_selector,
        result.linearized_collision.selector,
    )
    assert np.array_equal(
        result.placement_lift.guarded_state,
        result.placement_lift.guarded_lift.guarded_state,
    )
    placement_from_product = linearized_placement_result(
        state,
        result.lock_matrix,
        result.linearized_clear.clear_patterns,
        result.linearized_clear.selector,
        collision_selector=result.linearized_collision.selector,
        guarded_product=result.placement_lift.guarded_lift.product_state,
    )
    assert np.array_equal(result.next_state, placement_from_product.next_state)
    assert np.array_equal(
        result.next_state,
        result.placement_lift.state_matrix @ result.placement_lift.lifted_state,
    )
    assert result.linearized_clear.clear_lifted_state.shape == (
        expected_clear_lifted_size(candidate_count),
    )
    assert result.linearized_clear.clear_state_matrix.shape == (
        STATE_SIZE,
        expected_clear_lifted_size(candidate_count),
    )
    branch_count = 1 + (1 << candidate_count)
    assert result.placement_lift.selector.shape == (branch_count,)
    assert result.placement_lift.selector[0] == 0
    assert np.array_equal(result.placement_lift.selector[1:], result.linearized_clear.selector)
    assert result.placement_lift.guarded_state.shape == (branch_count * STATE_SIZE,)
    assert result.placement_lift.lifted_state.shape == (
        expected_placement_lifted_size(branch_count),
    )
    assert result.placement_lift.state_matrix.shape == (
        STATE_SIZE,
        expected_placement_lifted_size(branch_count),
    )
    assert np.array_equal(
        counts_from_q(q_from_state(result.next_state)),
        row_sum_matrix() @ direct_board,
    )
    return result


def demo() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.uint8)
    board[0, :6] = 1
    state = make_state(board)
    placement = Placement(piece="I", rotation=0, column=6)
    result = tetris_step(state, placement)

    print("Piecewise-linear Tetris demo")
    print(f"state dimension: {STATE_SIZE}")
    print(f"alive/dead: {int(result.next_state[ALIVE_INDEX])}/{int(result.next_state[DEAD_INDEX])}")
    print(f"L shape: {result.lock_matrix.shape}")
    print(f"K shape: {result.clear_matrix.shape}")
    print(f"hard-drop y values: {result.linearized_hard_drop.y_values}")
    print(f"hard-drop selector: {result.linearized_hard_drop.selector.tolist()}")
    print(f"hard-drop feature dimension: {result.linearized_hard_drop.feature_state.shape[0]}")
    print(f"collision selector: {result.linearized_collision.selector.tolist()}")
    print(
        "collision monomial-lift dimension: "
        f"{result.linearized_collision.occupancy_lift.lifted_state.shape[0]}"
    )
    print(
        "collision local tensor dimension: "
        f"{result.linearized_collision.local_occupancy_tensor.shape[0]}"
    )
    print(f"local clear rows: {result.linearized_clear.candidate_rows}")
    print(f"local clear patterns: {len(result.linearized_clear.clear_patterns)}")
    print(
        "clear count-lift dimension: "
        f"{result.linearized_clear.count_lift.lifted_state.shape[0]}"
    )
    print(f"clear selector: {result.linearized_clear.selector.tolist()}")
    print(f"clear lifted state dimension: {result.linearized_clear.clear_lifted_state.shape[0]}")
    print(f"clear state matrix shape: {result.linearized_clear.clear_state_matrix.shape}")
    print(
        "clear product-lift dimension: "
        f"{result.linearized_clear.guarded_lift.lifted_state.shape[0]}"
    )
    if result.linearized_clear.guarded_lift.quadratic_lift is not None:
        print(
            "clear quadratic-lift dimension: "
            f"{result.linearized_clear.guarded_lift.quadratic_lift.quadratic_state.shape[0]}"
        )
    print(f"guarded clear state dimension: {result.linearized_clear.guarded_state.shape[0]}")
    print(f"linearized clear matrix shape: {result.linearized_clear.clear_lift_matrix.shape}")
    print(f"placement branches: {result.placement_lift.branch_labels}")
    print(f"placement selector: {result.placement_lift.selector.tolist()}")
    if result.placement_lift.selector_lift is not None:
        print(
            "placement selector quadratic dimension: "
            f"{result.placement_lift.selector_lift.quadratic_state.shape[0]}"
        )
    print(
        "placement product-lift dimension: "
        f"{result.placement_lift.guarded_lift.lifted_state.shape[0]}"
    )
    if result.placement_lift.guarded_lift.quadratic_lift is not None:
        print(
            "placement quadratic-lift dimension: "
            f"{result.placement_lift.guarded_lift.quadratic_lift.quadratic_state.shape[0]}"
        )
    print(f"placement lifted state dimension: {result.placement_lift.lifted_state.shape[0]}")
    print(f"placement state matrix shape: {result.placement_lift.state_matrix.shape}")
    print(f"unified step-lift dimension: {result.step_lift.lifted_state.shape[0]}")
    print(f"unified step matrix shape: {result.step_lift.step_matrix.shape}")
    print(f"step feature-state dimension: {result.step_feature.feature_state.shape[0]}")
    print(f"feature-to-step matrix shape: {result.step_feature.feature_to_step_matrix.shape}")
    print(f"feature-step matrix shape: {result.step_feature.feature_step_matrix.shape}")
    family = action_matrix_family()
    branch_counts = [len(matrix.branches) for matrix in family.matrices]
    max_width = max(matrix.shape[1] for matrix in family.matrices)
    print(f"canonical action matrices: {len(family)}")
    print(f"action branch count range: {min(branch_counts)}..{max(branch_counts)}")
    print(f"widest lazy action matrix shape: ({STATE_SIZE}, {max_width})")
    degree_one_basis = MonomialBasis.from_coordinates(range(STATE_SIZE), max_degree=1)
    persistent = persistent_z_transition_for_linear_map(
        state,
        result.branch_matrix,
        degree_one_basis,
    )
    print(f"persistent z degree-1 dimension: {persistent.z_state.shape[0]}")
    print(f"persistent z transition shape: {persistent.transition_matrix.shape}")
    print(f"branch: y={result.y}, clear_rows={result.clear_rows}")
    print("\ninitial bottom rows:")
    print(render_bottom(board, rows=4))
    print("\nnext bottom rows:")
    print(render_bottom(board_from_state(result.next_state), rows=4))


def verify() -> None:
    board = np.zeros((ROWS, COLS), dtype=np.uint8)
    board[0, :6] = 1
    placement = Placement(piece="I", rotation=0, column=6)
    result = assert_live_step_matches_direct(board, placement, (0,))
    assert result.linearized_clear.selector.tolist() == [0, 1]
    assert_action_matrix_matches_step(make_state(board), placement)

    no_clear_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    no_clear_result = assert_live_step_matches_direct(
        no_clear_board,
        Placement(piece="O", rotation=0, column=4),
        (),
    )
    assert no_clear_result.linearized_clear.selector.tolist() == [1, 0, 0, 0]
    assert_action_matrix_matches_step(
        make_state(no_clear_board),
        Placement(piece="O", rotation=0, column=4),
    )

    two_line_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    two_line_board[0, :8] = 1
    two_line_board[1, :8] = 1
    two_line_placement = Placement(piece="O", rotation=0, column=8)
    two_line_result = assert_live_step_matches_direct(two_line_board, two_line_placement, (0, 1))
    assert len(two_line_result.linearized_clear.clear_patterns) == 4
    assert two_line_result.linearized_clear.selector.tolist() == [0, 0, 0, 1]

    four_line_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    four_line_board[0:4, :9] = 1
    four_line_result = assert_live_step_matches_direct(
        four_line_board,
        Placement(piece="I", rotation=1, column=9),
        (0, 1, 2, 3),
    )
    assert len(four_line_result.linearized_clear.clear_patterns) == 16
    assert four_line_result.linearized_clear.selector.tolist() == [0] * 15 + [1]
    assert not np.any(board_from_state(four_line_result.next_state))
    assert_action_matrix_matches_step(
        make_state(four_line_board),
        Placement(piece="I", rotation=1, column=9),
    )

    sparse_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    sparse_board[1, 1] = 1
    sparse_board[3, 3] = 1
    sparse_board[4, 4] = 1
    compacted = clear_matrix((0, 2)) @ make_state(sparse_board)
    expected_compacted = np.zeros((ROWS, COLS), dtype=np.uint8)
    expected_compacted[0, 1] = 1
    expected_compacted[1, 3] = 1
    expected_compacted[2, 4] = 1
    assert np.array_equal(board_from_state(compacted), expected_compacted.reshape(BOARD_SIZE))
    assert np.array_equal(
        counts_from_q(q_from_state(compacted)),
        row_sum_matrix() @ expected_compacted.reshape(BOARD_SIZE),
    )

    empty = np.zeros((ROWS, COLS), dtype=np.uint8)
    family = action_matrix_family()
    assert len(family) == CANONICAL_ACTION_COUNT
    assert len(family.placements) == CANONICAL_ACTION_COUNT
    assert len(set(family.placements)) == CANONICAL_ACTION_COUNT
    assert all(matrix.shape[0] == STATE_SIZE for matrix in family.matrices)
    assert all(matrix.shape[1] == len(matrix.branches) * STATE_SIZE for matrix in family.matrices)
    assert_action_matrix_matches_step(make_state(empty), family.placements[0])
    assert_action_matrix_matches_step(make_state(empty), family.placements[-1])

    for piece in sorted(BASE_SHAPES):
        for rotation, cells in enumerate(all_rotations(piece)):
            width = max(col for _, col in cells) + 1
            for column in (0, COLS - width):
                assert_live_step_matches_direct(
                    empty,
                    Placement(piece, rotation, column),
                    (),
                )

    blocker_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    blocker_board[0, 0] = 1
    blocker_step = assert_live_step_matches_direct(
        blocker_board,
        Placement("O", 0, 0),
        (),
    )
    assert blocker_step.y == 1
    assert blocker_step.linearized_hard_drop.selector.tolist()[0:3] == [0, 0, 1]

    no_landing_board = np.zeros((ROWS, COLS), dtype=np.uint8)
    no_landing_board[:, 0] = 1
    collision_step = tetris_step(make_state(no_landing_board), Placement("I", 1, 0))
    assert collision_step.died is True
    assert collision_step.y == -1
    assert collision_step.linearized_hard_drop.selector[0] == 1
    assert_action_matrix_matches_step(make_state(no_landing_board), Placement("I", 1, 0))
    assert is_dead_state(collision_step.next_state)
    assert_step_result_lift_consistent(collision_step)
    assert_persistent_degree_one_z_consistent(make_state(no_landing_board), collision_step)
    assert_collision_lift_consistent(collision_step.linearized_collision)
    assert collision_step.linearized_collision.selector.tolist() == [1, 0]
    assert_linearized_lock_consistent(collision_step.linearized_lock)
    assert np.array_equal(collision_step.linearized_lock.next_state, make_dead_state())
    assert np.array_equal(collision_step.next_state, make_dead_state())
    assert np.array_equal(
        collision_step.next_state,
        collision_step.branch_matrix @ make_state(no_landing_board),
    )
    assert np.array_equal(
        collision_step.next_state,
        collision_step.linearized_clear.clear_lift_matrix
        @ collision_step.linearized_clear.guarded_state,
    )
    assert_local_count_lift_consistent(collision_step.linearized_clear.count_lift)
    assert np.array_equal(
        collision_step.next_state,
        collision_step.linearized_clear.clear_state_matrix
        @ collision_step.linearized_clear.clear_lifted_state,
    )
    assert_guarded_lift_consistent(collision_step.linearized_clear.guarded_lift)
    assert collision_step.placement_lift.selector[0] == 1
    assert collision_step.placement_lift.selector_lift is None
    assert np.array_equal(
        collision_step.next_state,
        collision_step.placement_lift.branch_matrix @ collision_step.placement_lift.guarded_state,
    )
    assert_guarded_lift_consistent(collision_step.placement_lift.guarded_lift)
    assert np.array_equal(
        collision_step.next_state,
        collision_step.placement_lift.state_matrix @ collision_step.placement_lift.lifted_state,
    )

    for bad_placement in (
            Placement("I", 0, 7),
            Placement("O", 0, 9),
            Placement("O", 0, -1),
            Placement("X", 0, 0),
    ):
        bad_step = tetris_step(make_state(empty), bad_placement)
        assert bad_step.died is True
        assert bad_step.y == -1
        assert bad_step.linearized_hard_drop.selector.tolist() == [1]
        assert_action_matrix_matches_step(make_state(empty), bad_placement)
        assert np.array_equal(bad_step.next_state, make_dead_state())
        assert_step_result_lift_consistent(bad_step)
        assert_persistent_degree_one_z_consistent(make_state(empty), bad_step)
        assert_collision_lift_consistent(bad_step.linearized_collision)
        assert bad_step.linearized_collision.selector.tolist() == [1, 0]
        assert_linearized_lock_consistent(bad_step.linearized_lock)
        assert np.array_equal(bad_step.linearized_lock.next_state, make_dead_state())
        assert np.array_equal(bad_step.next_state, bad_step.branch_matrix @ make_state(empty))
        assert bad_step.placement_lift.selector.tolist() == [1]
        assert bad_step.placement_lift.selector_lift is None
        assert_local_count_lift_consistent(bad_step.linearized_clear.count_lift)
        assert_guarded_lift_consistent(bad_step.linearized_clear.guarded_lift)
        assert_guarded_lift_consistent(bad_step.placement_lift.guarded_lift)
        assert np.array_equal(
            bad_step.next_state,
            bad_step.placement_lift.state_matrix @ bad_step.placement_lift.lifted_state,
        )

    already_dead = make_dead_state()
    assert np.array_equal(lock_matrix(piece_mask("I", 0, 0, 0)) @ already_dead, already_dead)
    assert np.array_equal(clear_matrix(()) @ already_dead, already_dead)
    assert np.array_equal(clear_matrix((0,)) @ already_dead, already_dead)
    assert np.array_equal(death_matrix() @ already_dead, already_dead)
    absorbing_step = tetris_step(already_dead, Placement("I", 0, 0))
    assert absorbing_step.died is True
    assert absorbing_step.y == -1
    assert absorbing_step.linearized_hard_drop.selector.tolist() == [1]
    assert_action_matrix_matches_step(already_dead, Placement("I", 0, 0))
    assert is_dead_state(absorbing_step.next_state)
    assert_step_result_lift_consistent(absorbing_step)
    assert_persistent_degree_one_z_consistent(already_dead, absorbing_step)
    assert_collision_lift_consistent(absorbing_step.linearized_collision)
    assert absorbing_step.linearized_collision.selector.tolist() == [1, 0]
    assert_linearized_lock_consistent(absorbing_step.linearized_lock)
    assert np.array_equal(absorbing_step.linearized_lock.next_state, already_dead)
    assert np.array_equal(absorbing_step.next_state, already_dead)
    assert np.array_equal(absorbing_step.next_state, absorbing_step.branch_matrix @ already_dead)
    assert np.array_equal(
        absorbing_step.next_state,
        absorbing_step.linearized_clear.clear_state_matrix
        @ absorbing_step.linearized_clear.clear_lifted_state,
    )
    assert_guarded_lift_consistent(absorbing_step.linearized_clear.guarded_lift)
    assert_local_count_lift_consistent(absorbing_step.linearized_clear.count_lift)
    assert_guarded_lift_consistent(absorbing_step.placement_lift.guarded_lift)
    assert absorbing_step.placement_lift.selector_lift is None
    assert np.array_equal(
        absorbing_step.next_state,
        absorbing_step.placement_lift.state_matrix @ absorbing_step.placement_lift.lifted_state,
    )

    malformed_dead = make_dead_state()
    malformed_dead[HOMOG_INDEX] = 1
    assert_raises_value_error(lambda: as_state_vector(malformed_dead))

    malformed_dead = make_dead_state()
    malformed_dead[0] = 1
    assert_raises_value_error(lambda: as_state_vector(malformed_dead))

    malformed_live = make_state(empty)
    malformed_live[HOMOG_INDEX] = 0
    assert_raises_value_error(lambda: as_state_vector(malformed_live))

    malformed_flags = make_state(empty)
    malformed_flags[DEAD_INDEX] = 1
    assert_raises_value_error(lambda: as_state_vector(malformed_flags))

    print("verification passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate the augmented piecewise-linear Tetris update "
            "with promoted linear state for row clearing and shifting."
        )
    )
    parser.add_argument("--demo", action="store_true", help="print a one-line-clear example")
    parser.add_argument("--verify", action="store_true", help="run matrix/direct-update checks")
    parser.add_argument(
        "--closure",
        action="store_true",
        help="run a bounded monomial closure probe",
    )
    parser.add_argument(
        "--closure-actions",
        type=int,
        default=1,
        help="number of in-bounds fixed-y placements to include in the closure probe",
    )
    parser.add_argument(
        "--closure-max-features",
        type=int,
        default=20000,
        help="stop closure probing after this many features",
    )
    parser.add_argument(
        "--closure-max-degree",
        type=int,
        default=12,
        help="discard closure features above this monomial degree",
    )
    parser.add_argument(
        "--closure-max-processed",
        type=int,
        default=2000,
        help="stop closure probing after processing this many queued features",
    )
    parser.add_argument(
        "--action-family",
        action="store_true",
        help="print the canonical lazy action-matrix family summary",
    )
    parser.add_argument(
        "--action-count",
        type=int,
        default=CANONICAL_ACTION_COUNT,
        help="number of representative action matrices to build",
    )
    return parser.parse_args()


def closure_demo(args: argparse.Namespace) -> None:
    placements = all_in_bounds_placements()[: args.closure_actions]
    report = explore_closed_monomial_basis(
        placements,
        max_features=args.closure_max_features,
        max_degree=args.closure_max_degree,
        max_processed=args.closure_max_processed,
    )
    print("Closure probe")
    print(f"actions: {report.action_count}")
    print(f"branches: {report.branch_count}")
    print(f"processed features: {report.processed_features}")
    print(f"feature count: {report.feature_count}")
    print(f"max degree retained: {report.max_degree}")
    print(f"hit feature cap: {report.hit_feature_cap}")
    print(f"hit degree cap: {report.hit_degree_cap}")
    print(f"degree histogram: {report.degree_histogram}")


def action_family_demo(args: argparse.Namespace) -> None:
    family = action_matrix_family(args.action_count)
    branch_counts = [len(matrix.branches) for matrix in family.matrices]
    widths = [matrix.shape[1] for matrix in family.matrices]
    print("Action matrix family")
    print(f"matrices: {len(family)}")
    print(f"state dimension: {STATE_SIZE}")
    print(f"branch count range: {min(branch_counts)}..{max(branch_counts)}")
    print(f"matrix width range: {min(widths)}..{max(widths)}")
    print("first action:", family.placements[0])
    print("last action:", family.placements[-1])


def main() -> None:
    args = parse_args()
    if args.verify:
        verify()
    if args.closure:
        closure_demo(args)
    if args.action_family:
        action_family_demo(args)
    if args.demo or (not args.verify and not args.closure and not args.action_family):
        demo()


if __name__ == "__main__":
    main()
