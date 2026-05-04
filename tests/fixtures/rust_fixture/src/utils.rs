//! Helper utilities used by main / relative_user.

#[derive(Clone, Debug, Default)]
pub struct Counter {
    value: u32,
}

impl Counter {
    pub fn new() -> Self {
        Counter { value: 0 }
    }

    pub fn increment(&mut self, by: u32) -> u32 {
        self.value = helper(self.value + by);
        self.value
    }
}

pub fn helper(x: u32) -> u32 {
    x + 1
}

pub fn double(x: u32) -> u32 {
    helper(x) * 2
}
