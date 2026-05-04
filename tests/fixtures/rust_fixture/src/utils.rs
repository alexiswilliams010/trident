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

#[cfg(test)]
mod tests {
    //! Inline unit tests — should be filtered out of the semantic graph.

    use super::*;

    #[test]
    fn test_helper_increments() {
        assert_eq!(helper(0), 1);
    }

    #[test]
    fn test_double_doubles_helper() {
        assert_eq!(double(2), 6);
    }

    fn test_only_helper(x: u32) -> u32 {
        // Plain fn inside a cfg(test) module — also test-only, even
        // without #[test] on it.
        x + 100
    }
}
