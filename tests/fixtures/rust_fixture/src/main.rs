//! Binary entrypoint: exercises absolute imports against the current crate
//! and an external dependency.

use myapp::utils::{double, helper};
use serde::Serialize;

#[derive(Serialize)]
pub struct Report {
    pub total: u32,
}

pub fn run() -> u32 {
    let a = helper(1);
    let b = double(2);
    a + b
}

fn main() {
    let _ = run();
}
