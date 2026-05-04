//! Exercises `super::` relative imports — Phase 3 should resolve this to
//! src/utils.rs in the same crate.

use super::utils::helper;

pub fn relative_call(x: u32) -> u32 {
    helper(x)
}
