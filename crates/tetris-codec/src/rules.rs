//! Occupancy invariant shared by symbolic construction and admission diagnostics.
//! Keep single-use helpers at their call sites and file-local helpers in their owning module.
/// Returns the cell-count modulus preserved by placements and line clears.
pub const fn occupancy_modulus(cols: usize) -> usize {
    if cols.is_multiple_of(4) {
        4
    } else if cols.is_multiple_of(2) {
        2
    } else {
        1
    }
}
