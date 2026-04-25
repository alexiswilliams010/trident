"""File walker exclusion logic: --exclude flags and .tsgrepignore."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.file_walker import (
    TSGREP_IGNORE_FILE,
    WalkConfig,
    _matches_excludes,
    read_tsgrepignore,
    walk_repo,
)


# ────────────────────────────────────────────────────────────────────
# Pattern matching
# ────────────────────────────────────────────────────────────────────


def test_basename_pattern_matches_any_component():
    # No-slash pattern → match any path component.
    assert _matches_excludes("src/test/Foo.sol", ("test",))
    assert _matches_excludes("test/Foo.sol", ("test",))
    assert not _matches_excludes("src/Foo.sol", ("test",))


def test_glob_basename_matches_filename():
    assert _matches_excludes("src/MyTest.t.sol", ("*.t.sol",))
    assert _matches_excludes("test/Foo.t.sol", ("*.t.sol",))
    assert not _matches_excludes("src/MyTest.sol", ("*.t.sol",))


def test_path_pattern_with_slash_matches_full_relpath():
    # Pattern with `/` → full-path fnmatch.
    assert _matches_excludes("src/legacy/Foo.sol", ("src/legacy/*",))
    # The same pattern won't match deeper-nested files because fnmatch's `*`
    # is greedy enough to cross slashes (single-level here is enough).
    assert _matches_excludes("src/legacy/sub/Bar.sol", ("src/legacy/*",))
    assert not _matches_excludes("src/main/Foo.sol", ("src/legacy/*",))


def test_empty_patterns_match_nothing():
    assert not _matches_excludes("any/path.py", ())


# ────────────────────────────────────────────────────────────────────
# .tsgrepignore parsing
# ────────────────────────────────────────────────────────────────────


def test_read_tsgrepignore_skips_blanks_and_comments(tmp_path: Path):
    (tmp_path / TSGREP_IGNORE_FILE).write_text(
        "# comment\n\ntest\n*.t.sol\nscript/\n   \n# another comment\n"
    )
    patterns = read_tsgrepignore(tmp_path)
    # Trailing slash on `script/` should be stripped.
    assert patterns == ("test", "*.t.sol", "script")


def test_read_tsgrepignore_returns_empty_when_missing(tmp_path: Path):
    assert read_tsgrepignore(tmp_path) == ()


# ────────────────────────────────────────────────────────────────────
# walk_repo with exclude_patterns
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """Repo with src/ and test/ Solidity files plus a Python helper."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Vault.sol").write_text("contract Vault {}")
    (tmp_path / "src" / "MyTest.t.sol").write_text("contract MyTest {}")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "Vault.t.sol").write_text("contract VaultTest {}")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "Deploy.s.sol").write_text("contract Deploy {}")
    (tmp_path / "main.py").write_text("def f(): pass")
    return tmp_path


def test_walk_repo_no_excludes_yields_everything(tiny_repo: Path):
    cfg = WalkConfig.with_defaults(tiny_repo)
    rels = sorted(f.rel_path for f in walk_repo(cfg))
    assert rels == [
        "main.py",
        "scripts/Deploy.s.sol",
        "src/MyTest.t.sol",
        "src/Vault.sol",
        "test/Vault.t.sol",
    ]


def test_walk_repo_prunes_excluded_directory(tiny_repo: Path):
    cfg = WalkConfig.with_defaults(tiny_repo)
    cfg.exclude_patterns = ("test",)  # basename match prunes the `test/` dir
    rels = sorted(f.rel_path for f in walk_repo(cfg))
    assert "test/Vault.t.sol" not in rels
    assert "src/Vault.sol" in rels


def test_walk_repo_excludes_files_by_glob(tiny_repo: Path):
    cfg = WalkConfig.with_defaults(tiny_repo)
    cfg.exclude_patterns = ("*.t.sol", "*.s.sol")
    rels = sorted(f.rel_path for f in walk_repo(cfg))
    # Both Foundry test files and the deploy script are gone.
    assert "src/MyTest.t.sol" not in rels
    assert "test/Vault.t.sol" not in rels
    assert "scripts/Deploy.s.sol" not in rels
    # But the real source survives.
    assert "src/Vault.sol" in rels


def test_walk_repo_combines_path_and_basename_patterns(tiny_repo: Path):
    cfg = WalkConfig.with_defaults(tiny_repo)
    cfg.exclude_patterns = ("scripts", "*.t.sol")
    rels = sorted(f.rel_path for f in walk_repo(cfg))
    assert rels == ["main.py", "src/Vault.sol"]
