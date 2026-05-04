use super::super::runner;  // exercises chained `super::super`

pub fn nested_helper(x: u32) -> u32 {
    let _ = runner::run;
    x + 7
}
