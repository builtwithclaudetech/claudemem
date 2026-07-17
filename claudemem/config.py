"""claudemem.config — config parsing + scope resolution (L1, leaf).

This module is the leaf of the dependency graph (architecture §2.1, §3.3): it
depends on the **standard library only** and must NOT open SQLite, touch the
network, or import ``enrich``/``anthropic``/``recall``/``store``/``files``. That
absence is what keeps ``config`` trivially importable on the SC-2 cold path and
on the read-path side of the SC-6 firewall.

Responsibilities (architecture §2.1):

* ``load_config`` — parse ``$CLAUDEMEM_HOME/config.toml`` with stdlib ``tomllib``
  (binary read mode), applying every locked default from tech-design §9 when the
  file or an individual key is absent (PRD IN-18 done-condition, C-15).
* ``resolve_scope`` — derive the active project scope from ``cwd`` (PRD AS-8,
  Q-6, IN-3), overridable by an explicit ``--scope`` / ``--project`` flag.

All defaults below are reproduced as named module constants so they are
auditable against tech-design §9.

**Home-dir override mechanism.** The claudemem home directory (where
``config.toml`` and the DB files live) is read from the ``CLAUDEMEM_HOME``
environment variable, defaulting to ``~/.claude/claudemem`` (C-15). Tests point
``CLAUDEMEM_HOME`` at a ``tmp_path`` so they never touch the real ``~/.claude``
and run cleanly in fresh interpreters (consistent with how the firewall /
recursion tests spawn subprocesses). ``load_config`` also accepts an explicit
``config_path`` for callers that want to bypass the env var entirely.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

# --------------------------------------------------------------------------- #
# Locked defaults — reproduced verbatim from tech-design §9. Every value here   #
# is the authority for the IN-18 "default applied when absent" done-condition.  #
# --------------------------------------------------------------------------- #

# [llm] (tech-design §9; §5.4/§5.6/§5.7/§5.9)
LLM_BACKEND_DEFAULT = "auto"
LLM_MODEL_DEFAULT = "haiku"
LLM_CLI_CHUNK_SIZE_DEFAULT = 25
LLM_CLI_MAX_OUTPUT_TOKENS_DEFAULT = 16000  # raised 8000→16000 per MF-2 (§5.7)
LLM_CLI_PARSE_RETRIES_DEFAULT = 1
LLM_DEDUP_K_DEFAULT = 5
LLM_DEDUP_EXCERPT_CHARS_DEFAULT = 500

# [ranking] (tech-design §9; §4.1/§4.4/§4.5)
RANKING_SALIENCE_FLOOR_DEFAULT = 0.05
RANKING_RELEVANCE_FLOOR_DEFAULT = 0.30
RANKING_RECENCY_HALF_LIFE_DAYS_DEFAULT = 90
RANKING_IMPORTANCE_CURVE_DEFAULT = "normalized"
RANKING_CANDIDATE_K_DEFAULT = 64
RANKING_BM25_NAME_WEIGHT_DEFAULT = 10.0
RANKING_BM25_SUMMARY_WEIGHT_DEFAULT = 5.0
RANKING_BM25_ALIASES_WEIGHT_DEFAULT = 8.0
RANKING_BM25_BODY_WEIGHT_DEFAULT = 1.0

# [staleness] (tech-design §9; SC-13, IN-16, IN-10)
STALENESS_HORIZON_DAYS_DEFAULT = 180  # disuse beyond this ⇒ stale trust flag (decay-of-trust, distinct from [ranking] recency)

# [forkb] (tech-design §9; §3.5)
FORKB_WINDOW_DAYS_DEFAULT = 45
FORKB_ENTRY_CHAR_CAP_DEFAULT = 4000
FORKB_TRUNCATION_DEFAULT = "head_tail"

# [spend] (tech-design §9; §3.4/§5.10)
SPEND_DAILY_TOKEN_CAP_DEFAULT = 1_000_000
SPEND_MONTHLY_TOKEN_CAP_DEFAULT = 15_000_000
SPEND_CLI_DAILY_RECORD_CAP_DEFAULT = 2000
SPEND_CLI_DAILY_SPAWN_CAP_DEFAULT = 200
SPEND_WARN_FRACTION_DEFAULT = 0.8
SPEND_WINDOW_TZ_DEFAULT = "America/New_York"

# [promotion] (tech-design §9; IN-15)
PROMOTION_HIT_THRESHOLD_DEFAULT = 3
PROMOTION_WINDOW_DAYS_DEFAULT = 45

# [menu] (tech-design §9; SC-5, IN-11)
MENU_MAX_ENTRIES_DEFAULT = 30
MENU_TOKEN_CEILING_DEFAULT = 600

# --------------------------------------------------------------------------- #
# Path constants                                                                #
# --------------------------------------------------------------------------- #

#: Default claudemem home dir when ``$CLAUDEMEM_HOME`` is unset (C-15, §8.3).
DEFAULT_HOME = Path("~/.claude/claudemem").expanduser()
#: Fork A global-scope memory dir (architecture §8.3): always ``~/.claude/memory``.
GLOBAL_MEMORY_DIR = Path("~/.claude/memory").expanduser()
#: Root under which Claude Code stores per-project data (the cwd-slug convention).
PROJECTS_ROOT = Path("~/.claude/projects").expanduser()
#: Config file name within the claudemem home dir.
CONFIG_FILENAME = "config.toml"

CONFIG_HOME_ENV = "CLAUDEMEM_HOME"

#: Any run of characters Claude Code does not keep in a project slug.
_SLUG_NON_KEPT = re.compile(r"[^A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Typed, frozen settings                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LlmSettings:
    backend: str = LLM_BACKEND_DEFAULT
    model: str = LLM_MODEL_DEFAULT
    cli_chunk_size: int = LLM_CLI_CHUNK_SIZE_DEFAULT
    cli_max_output_tokens: int = LLM_CLI_MAX_OUTPUT_TOKENS_DEFAULT
    cli_parse_retries: int = LLM_CLI_PARSE_RETRIES_DEFAULT
    dedup_k: int = LLM_DEDUP_K_DEFAULT
    dedup_excerpt_chars: int = LLM_DEDUP_EXCERPT_CHARS_DEFAULT


@dataclass(frozen=True, slots=True)
class RankingSettings:
    salience_floor: float = RANKING_SALIENCE_FLOOR_DEFAULT
    relevance_floor: float = RANKING_RELEVANCE_FLOOR_DEFAULT
    recency_half_life_days: int = RANKING_RECENCY_HALF_LIFE_DAYS_DEFAULT
    importance_curve: str = RANKING_IMPORTANCE_CURVE_DEFAULT
    candidate_k: int = RANKING_CANDIDATE_K_DEFAULT
    bm25_name_weight: float = RANKING_BM25_NAME_WEIGHT_DEFAULT
    bm25_summary_weight: float = RANKING_BM25_SUMMARY_WEIGHT_DEFAULT
    bm25_aliases_weight: float = RANKING_BM25_ALIASES_WEIGHT_DEFAULT
    bm25_body_weight: float = RANKING_BM25_BODY_WEIGHT_DEFAULT


@dataclass(frozen=True, slots=True)
class StalenessSettings:
    horizon_days: int = STALENESS_HORIZON_DAYS_DEFAULT


@dataclass(frozen=True, slots=True)
class ForkbSettings:
    window_days: int = FORKB_WINDOW_DAYS_DEFAULT
    entry_char_cap: int = FORKB_ENTRY_CHAR_CAP_DEFAULT
    truncation: str = FORKB_TRUNCATION_DEFAULT


@dataclass(frozen=True, slots=True)
class SpendSettings:
    daily_token_cap: int = SPEND_DAILY_TOKEN_CAP_DEFAULT
    monthly_token_cap: int = SPEND_MONTHLY_TOKEN_CAP_DEFAULT
    cli_daily_record_cap: int = SPEND_CLI_DAILY_RECORD_CAP_DEFAULT
    cli_daily_spawn_cap: int = SPEND_CLI_DAILY_SPAWN_CAP_DEFAULT
    warn_fraction: float = SPEND_WARN_FRACTION_DEFAULT
    window_tz: str = SPEND_WINDOW_TZ_DEFAULT


@dataclass(frozen=True, slots=True)
class PromotionSettings:
    hit_threshold: int = PROMOTION_HIT_THRESHOLD_DEFAULT
    window_days: int = PROMOTION_WINDOW_DAYS_DEFAULT


@dataclass(frozen=True, slots=True)
class MenuSettings:
    max_entries: int = MENU_MAX_ENTRIES_DEFAULT
    token_ceiling: int = MENU_TOKEN_CEILING_DEFAULT


@dataclass(frozen=True, slots=True)
class Settings:
    """The fully-resolved, frozen configuration (tech-design §9)."""

    llm: LlmSettings = LlmSettings()
    ranking: RankingSettings = RankingSettings()
    staleness: StalenessSettings = StalenessSettings()
    forkb: ForkbSettings = ForkbSettings()
    spend: SpendSettings = SpendSettings()
    promotion: PromotionSettings = PromotionSettings()
    menu: MenuSettings = MenuSettings()


@dataclass(frozen=True, slots=True)
class ScopeContext:
    """The active memory scope for an invocation (architecture §2.1; AS-8/Q-6/IN-3).

    ``kind`` is ``"global"`` or ``"project"``. ``project_id`` is the cwd-derived
    Claude Code project slug (``None`` for global scope). ``global_dir`` /
    ``project_dir`` are the resolved Fork A memory directories; ``project_dir``
    is ``None`` for global scope.
    """

    kind: str
    project_id: str | None
    global_dir: Path
    project_dir: Path | None


# --------------------------------------------------------------------------- #
# load_config                                                                   #
# --------------------------------------------------------------------------- #


def _home_dir() -> Path:
    """Resolve the claudemem home dir from ``$CLAUDEMEM_HOME`` or the default."""
    override = os.environ.get(CONFIG_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_HOME


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(
    overrides: dict[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> Settings:
    """Load ``config.toml`` and return a frozen :class:`Settings` (C-15, IN-18).

    Resolution order for the config file: ``config_path`` (explicit) →
    ``$CLAUDEMEM_HOME/config.toml`` → ``~/.claude/claudemem/config.toml``. An
    absent file yields all locked defaults; an absent key within a present file
    yields that key's locked default (the IN-18 done-condition).

    ``overrides`` is an optional mapping of ``{section: {key: value}}`` applied
    on top of file values (used by tests and programmatic callers); unknown
    sections/keys in either source are ignored so a stray key never errors
    (SC-3 posture).
    """
    path = config_path if config_path is not None else _home_dir() / CONFIG_FILENAME

    raw: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as fh:
            raw = tomllib.load(fh)

    if overrides:
        merged: dict[str, Any] = {
            k: dict(v) for k, v in raw.items() if isinstance(v, dict)
        }
        for section, values in overrides.items():
            if isinstance(values, dict):
                merged.setdefault(section, {}).update(values)
        raw = merged

    return Settings(
        llm=_build(LlmSettings, _section(raw, "llm")),
        ranking=_build(RankingSettings, _section(raw, "ranking")),
        staleness=_build(StalenessSettings, _section(raw, "staleness")),
        forkb=_build(ForkbSettings, _section(raw, "forkb")),
        spend=_build(SpendSettings, _section(raw, "spend")),
        promotion=_build(PromotionSettings, _section(raw, "promotion")),
        menu=_build(MenuSettings, _section(raw, "menu")),
    )


_T = TypeVar("_T")


def _build(cls: type[_T], values: dict[str, Any]) -> _T:
    """Construct a frozen settings dataclass, applying only known keys.

    Each dataclass field already carries its locked default, so any key absent
    from ``values`` falls back to that default (IN-18). Unknown keys are ignored
    rather than raising, so a forward-compatible config never errors (SC-3).
    """
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    known = {k: v for k, v in values.items() if k in fields}
    return cls(**known)


# --------------------------------------------------------------------------- #
# resolve_scope                                                                 #
# --------------------------------------------------------------------------- #


def _project_id_from_cwd(cwd: Path) -> str:
    """Derive the Claude Code project slug from an absolute cwd (AS-8/Q-6).

    Matches Claude Code's own project-to-directory mapping: every run of
    non-alphanumeric characters in the absolute path (directory separators AND
    spaces, dots, etc.) collapses to a single ``-`` (e.g.
    ``/home/you/projects/my-app`` →
    ``-home-you-projects-my-app``). The per-project memory dir is
    ``~/.claude/projects/<slug>/memory`` (architecture §8.3); this matches the
    real on-disk slug Claude Code uses for this very repo.
    """
    return _SLUG_NON_KEPT.sub("-", str(cwd))


def resolve_scope(
    cwd: Path,
    project_flag: str | None = None,
    scope_flag: str | None = None,
) -> ScopeContext:
    """Resolve the active memory scope (architecture §2.1; AS-8, Q-6, IN-3).

    The scope is cwd-derived by default; an explicit ``--project`` (a project id
    / slug) or ``--scope`` flag overrides it (PRD IN-3/IN-18):

    * ``scope_flag == "global"`` → global scope (``project_id`` is ``None``).
    * ``scope_flag == "project"`` → project scope; the project id comes from
      ``project_flag`` if given, else from ``cwd``.
    * ``project_flag`` given (without a conflicting ``scope_flag``) → project
      scope on that explicit id.
    * neither flag → project scope derived from ``cwd``.

    ``global_dir`` is always ``~/.claude/memory``; ``project_dir`` is
    ``~/.claude/projects/<project_id>/memory`` and is ``None`` for global scope.
    """
    if scope_flag == "global":
        return ScopeContext(
            kind="global",
            project_id=None,
            global_dir=GLOBAL_MEMORY_DIR,
            project_dir=None,
        )

    project_id = project_flag if project_flag else _project_id_from_cwd(cwd)
    return ScopeContext(
        kind="project",
        project_id=project_id,
        global_dir=GLOBAL_MEMORY_DIR,
        project_dir=PROJECTS_ROOT / project_id / "memory",
    )


__all__ = [
    "Settings",
    "LlmSettings",
    "RankingSettings",
    "StalenessSettings",
    "ForkbSettings",
    "SpendSettings",
    "PromotionSettings",
    "MenuSettings",
    "ScopeContext",
    "load_config",
    "resolve_scope",
]
