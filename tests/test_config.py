"""Tests for claudemem.config (T0.4).

Covers the IN-18 done-condition exhaustively: every tech-design §9 key, with an
absent config file and with an empty ``config.toml``, applies its locked
default. Plus partial-override, non-default honoring, and the AS-8/Q-6/IN-3
scope-resolution paths.

The claudemem home dir is pointed at ``tmp_path`` via the ``CLAUDEMEM_HOME``
env var so no test ever touches the real ``~/.claude``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claudemem import config
from claudemem.config import Settings, load_config, resolve_scope


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it (no config.toml written)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


# Every §9 key, section by section, as (section, attr, expected-default).
# Reproduced from tech-design §9; this list IS the IN-18 done-condition coverage.
LOCKED_DEFAULTS: list[tuple[str, str, object]] = [
    # [llm]
    ("llm", "backend", "auto"),
    ("llm", "model", "haiku"),
    ("llm", "cli_chunk_size", 25),
    ("llm", "cli_max_output_tokens", 16000),  # MF-2: 16000, not 8000
    ("llm", "cli_parse_retries", 1),
    ("llm", "dedup_k", 5),
    ("llm", "dedup_excerpt_chars", 500),
    # [ranking]
    ("ranking", "salience_floor", 0.05),
    ("ranking", "relevance_floor", 0.30),
    ("ranking", "recency_half_life_days", 90),
    ("ranking", "importance_curve", "normalized"),
    ("ranking", "candidate_k", 64),
    ("ranking", "bm25_name_weight", 10.0),
    ("ranking", "bm25_summary_weight", 5.0),
    ("ranking", "bm25_aliases_weight", 8.0),
    ("ranking", "bm25_body_weight", 1.0),
    # [forkb]
    ("forkb", "window_days", 45),
    ("forkb", "entry_char_cap", 4000),
    ("forkb", "truncation", "head_tail"),
    # [spend]
    ("spend", "daily_token_cap", 1_000_000),
    ("spend", "monthly_token_cap", 15_000_000),
    ("spend", "cli_daily_record_cap", 2000),
    ("spend", "cli_daily_spawn_cap", 200),
    ("spend", "warn_fraction", 0.8),
    ("spend", "window_tz", "America/New_York"),
    # [promotion]
    ("promotion", "hit_threshold", 3),
    ("promotion", "window_days", 45),
    # [menu]
    ("menu", "max_entries", 30),
    ("menu", "token_ceiling", 600),
]


def _get(settings: Settings, section: str, attr: str) -> object:
    return getattr(getattr(settings, section), attr)


@pytest.mark.parametrize(("section", "attr", "expected"), LOCKED_DEFAULTS)
def test_absent_file_applies_default(
    home: Path, section: str, attr: str, expected: object
) -> None:
    """With no config.toml at all, every §9 key is its locked default (IN-18)."""
    assert not (home / config.CONFIG_FILENAME).exists()
    settings = load_config()
    assert _get(settings, section, attr) == expected


@pytest.mark.parametrize(("section", "attr", "expected"), LOCKED_DEFAULTS)
def test_empty_file_applies_default(
    home: Path, section: str, attr: str, expected: object
) -> None:
    """With an empty config.toml, every §9 key is its locked default (IN-18)."""
    (home / config.CONFIG_FILENAME).write_text("", encoding="utf-8")
    settings = load_config()
    assert _get(settings, section, attr) == expected


def test_default_count_matches_field_count() -> None:
    """Guard: LOCKED_DEFAULTS must cover every field of every settings section.

    Catches a future §9 key added to a dataclass without a test row.
    """
    section_classes = {
        "llm": config.LlmSettings,
        "ranking": config.RankingSettings,
        "forkb": config.ForkbSettings,
        "spend": config.SpendSettings,
        "promotion": config.PromotionSettings,
        "menu": config.MenuSettings,
    }
    covered: set[tuple[str, str]] = {(s, a) for s, a, _ in LOCKED_DEFAULTS}
    for section, cls in section_classes.items():
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            assert (section, field_name) in covered, (section, field_name)


def test_partial_override_leaves_others_at_default(home: Path) -> None:
    """A partial config.toml overriding a few keys leaves all others default."""
    (home / config.CONFIG_FILENAME).write_text(
        '[llm]\nbackend = "cli"\ncli_chunk_size = 10\n\n[menu]\nmax_entries = 12\n',
        encoding="utf-8",
    )
    settings = load_config()

    # Overridden keys honored.
    assert settings.llm.backend == "cli"
    assert settings.llm.cli_chunk_size == 10
    assert settings.menu.max_entries == 12

    # A sibling key in a touched section stays default.
    assert settings.llm.model == "haiku"
    assert settings.llm.cli_max_output_tokens == 16000
    assert settings.menu.token_ceiling == 600

    # An entirely untouched section stays default.
    assert settings.ranking.relevance_floor == 0.30
    assert settings.spend.daily_token_cap == 1_000_000


def test_non_default_value_is_honored(home: Path) -> None:
    """A provided key with a non-default value is read back as given."""
    (home / config.CONFIG_FILENAME).write_text(
        "[ranking]\nrelevance_floor = 0.5\nrecency_half_life_days = 30\n",
        encoding="utf-8",
    )
    settings = load_config()
    assert settings.ranking.relevance_floor == 0.5
    assert settings.ranking.recency_half_life_days == 30


def test_unknown_key_is_ignored(home: Path) -> None:
    """A stray/forward-compat key never errors (SC-3 posture)."""
    (home / config.CONFIG_FILENAME).write_text(
        '[llm]\nbackend = "sdk"\nfuture_knob = 99\n\n[mystery]\nx = 1\n',
        encoding="utf-8",
    )
    settings = load_config()
    assert settings.llm.backend == "sdk"


def test_overrides_arg_applied_over_file(home: Path) -> None:
    """The overrides mapping wins over file values and the defaults."""
    (home / config.CONFIG_FILENAME).write_text(
        '[llm]\nbackend = "cli"\n', encoding="utf-8"
    )
    settings = load_config(overrides={"llm": {"backend": "sdk", "dedup_k": 9}})
    assert settings.llm.backend == "sdk"
    assert settings.llm.dedup_k == 9
    assert settings.menu.max_entries == 30  # untouched → default


def test_explicit_config_path_bypasses_env(tmp_path: Path) -> None:
    """An explicit config_path is read regardless of CLAUDEMEM_HOME."""
    cfg = tmp_path / "custom.toml"
    cfg.write_text("[menu]\ntoken_ceiling = 123\n", encoding="utf-8")
    settings = load_config(config_path=cfg)
    assert settings.menu.token_ceiling == 123


def test_settings_is_frozen(home: Path) -> None:
    """Settings (and its sub-sections) are immutable."""
    settings = load_config()
    with pytest.raises(AttributeError):
        settings.llm.backend = "sdk"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        settings.menu = config.MenuSettings()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# resolve_scope (AS-8 / Q-6 / IN-3)                                             #
# --------------------------------------------------------------------------- #


def test_scope_cwd_derived_project() -> None:
    """No flags → project scope derived from cwd (the Claude Code slug)."""
    cwd = Path("/home/you/projects/my-app")
    ctx = resolve_scope(cwd)
    assert ctx.kind == "project"
    assert ctx.project_id == "-home-you-projects-my-app"
    assert ctx.global_dir == config.GLOBAL_MEMORY_DIR
    assert ctx.project_dir == (
        config.PROJECTS_ROOT / "-home-you-projects-my-app" / "memory"
    )


def test_scope_global_flag() -> None:
    """--scope global → global scope, no project id / dir."""
    ctx = resolve_scope(Path("/home/you/anything"), scope_flag="global")
    assert ctx.kind == "global"
    assert ctx.project_id is None
    assert ctx.project_dir is None
    assert ctx.global_dir == config.GLOBAL_MEMORY_DIR


def test_scope_project_flag_overrides_cwd() -> None:
    """--project <id> beats the cwd-derived id."""
    cwd = Path("/home/you/projects/my-app")
    ctx = resolve_scope(cwd, project_flag="-other-project")
    assert ctx.kind == "project"
    assert ctx.project_id == "-other-project"
    assert ctx.project_dir == config.PROJECTS_ROOT / "-other-project" / "memory"


def test_scope_flag_project_with_explicit_id() -> None:
    """--scope project together with --project pins the explicit id."""
    cwd = Path("/home/you/whatever")
    ctx = resolve_scope(cwd, project_flag="-pinned", scope_flag="project")
    assert ctx.kind == "project"
    assert ctx.project_id == "-pinned"


def test_scope_flag_project_without_id_uses_cwd() -> None:
    """--scope project without --project falls back to the cwd-derived id."""
    cwd = Path("/home/you/repo")
    ctx = resolve_scope(cwd, scope_flag="project")
    assert ctx.kind == "project"
    assert ctx.project_id == "-home-you-repo"


def test_scope_context_is_frozen() -> None:
    """ScopeContext is immutable."""
    ctx = resolve_scope(Path("/x"))
    with pytest.raises(AttributeError):
        ctx.kind = "global"  # type: ignore[misc]
