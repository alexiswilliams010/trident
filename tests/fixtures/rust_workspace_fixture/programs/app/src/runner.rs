//! Cross-crate import: `core_lib` is a workspace sibling, not external.

use core_lib::helpers::shared;
use core_lib::state::Counter;
use crate::nested::deep::nested_helper;
use serde::Serialize;

#[derive(Serialize)]
pub struct Report {
    pub total: u32,
}

pub fn run() -> u32 {
    let c = Counter::default();
    let nested = nested_helper(c.value);
    shared(nested + 1)
}
