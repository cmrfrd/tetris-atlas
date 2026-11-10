# Exact reachable single-row codec

For `BoardCodec<C, 1, H>` with 8–12 columns and the full seven-piece bag, exactly `C::COLS - 2` boards
are reachable from the empty/full-bag seed while every post-clear board stays at height <=1.
For ten columns this is exactly eight boards. The hole budget H is irrelevant: a one-row board has no holes.

On empty, only a horizontal I can leave height <=1. No piece has enough cells to clear a row of at least eight cells from empty. Every other first placement therefore exceeds the cap. A horizontal I has `C::COLS - 3` possible columns, yielding those four-cell boards plus empty.

After that I, it is absent from the remaining bag. Each of the other six pieces is at least two
rows tall and has at most three cells in a row. Alongside the four existing cells it cannot fill a row of at least eight cells, and it cannot fill any higher row that was empty.
It therefore clears no rows and leaves height >=2. No second move stays within the cap, and a second I cannot be drawn before
the bag is exhausted. This proves both necessity and completeness of this carrier (eight boards for ten columns).

Reachability here projects legal board/bag histories onto boards; it does not certify arbitrary
remaining-bag values attached separately to an ID.

This is a history cap, not just a target-board height restriction. Boards produced by earlier
height excursions are outside this specialization's contract. It is not a winning policy:
some draws immediately force departure from this tiny carrier.

## Arithmetic rank and select

The empty row has ID zero. A nonempty row is admitted exactly when it is four consecutive bits
starting at column `0..=C::COLS-4`. Its ID is `1 + trailing_zeros(row)`. Selection returns
`0b1111 << (id - 1)` for IDs `1..=C::COLS-3`. There is no board catalog, excluded-ID table, generated graph,
heap allocation, or runtime search. Public native-board and row entry points share this rule.

`GameDimensions` derives a `FULL_BAG` boolean from the typed `C::PIECE_SET` value.
This prevents a full-seven-bag proof being applied to an I-only game: two consecutive horizontal
I placements are legal in that game and their eight-cell board remains admitted by its existing
conservative codec. Other piece sets retain their previous necessary-condition language.

Native integer selection also supports `u8` for ranges up to 256 IDs; the eight-state codec
needs three packed bits or one byte per native ID. A one-byte-per-state payload needs eight bytes,
versus 134 before. Older numeric single-row datasets need reindexing; there is no old-ID decoder.

## Independent verification

Module-local tests compute the complete forward engine closure including the remaining bag,
without filtering expansion through the codec. Every one of 65,536 row masks is compared with
that independently reached set, including invalid high bits. All 1,024 native one-row boards
are checked too. Every ID is round-tripped through native/row and typed/untyped APIs. Fixed seeded
`TetrisGame` plays independently witness all seven nonempty boards. An actual I-only game verifies
that the bag-dependent exclusion cannot leak into other configured piece sets.

The unrestricted-piece-order negative control reaches 41 boards at the same height. It is not
the 7-bag game. At height two, an exhaustive unrestricted-order engine search reaches 183,002
boards; cumulative targets with holes 0..4 number 20,008 / 66,132 / 118,182 / 157,332 / 175,858.
Those are upper bounds for legal seven-bag reachability, not production codec counts. No height-two
catalog is embedded: algebraic ranking and bag-sensitive proof remain unresolved there.

Ten-column height caps 4..6 likewise retain their previous arithmetic/graph rules. Their remaining
boards have not all been proved seed-reachable. Sampling absence is never used to exclude a board.

A regression plays a seeded I/T/L prefix on disjoint floor intervals: it reaches height two,
clears the bottom row and leaves two cells at height one. The one-row codec rejects that board,
while height caps two and four through six admit it. Applying the one-row proof merely because
a larger codec receives a low target board would be unsound; specialization dispatch uses the
configured whole-history cap.

The same proof and independent full-bag engine closures cover all supported widths 8, 9, 10, 11 and 12; their state counts are 6, 7, 8, 9 and 10 respectively.
