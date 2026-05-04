//! Integration test — cargo treats `tests/` as a separate compilation
//! unit, so this file should be skipped at the file-path level by the
//! semantic resolver.

use myapp::utils::helper;

#[test]
fn test_integration_helper() {
    assert_eq!(helper(41), 42);
}
