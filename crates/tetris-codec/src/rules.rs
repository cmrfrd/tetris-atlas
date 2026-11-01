//! Occupancy invariant shared by symbolic construction and admission diagnostics.
//! Keep single-use helpers at their call sites and file-local helpers in their owning module.
pub const fn occupancy_modulus(cols: usize) -> usize {
    if cols % 4 == 0 {
        4
    } else if cols % 2 == 0 {
        2
    } else {
        1
    }
}
