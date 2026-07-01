# Phase 7 — Repository Cleanup & Architecture (Teaching Notes)

Phase 7 is where a working prototype becomes a **GitHub-ready** project. Nothing
here changes what the model does; it changes how maintainable, understandable, and
trustworthy the codebase is.

## Clean architecture: the `src/` package

Everything lives under `src/pitgpt/` as an installable package (`pip install -e .`
via `pyproject.toml`). Benefits:

- `import pitgpt...` works identically in scripts, tests, and the app — no
  `sys.path` hacks.
- The `src/` layout stops Python from accidentally importing the *repo folder*
  instead of the *installed package*, which catches "works on my machine" bugs.
- Clear module boundaries mirror the pipeline: `config → data → training →
  inference/evaluation → app`.

## Configuration management (single source of truth)

`config/settings.py` holds every path, model id, and hyperparameter as frozen
dataclasses. No other module hard-codes a magic string. Changing the LoRA rank,
adapter directory, or decoding temperature is a one-line edit that the whole
pipeline picks up. Frozen dataclasses turn typos into loud `AttributeError`s
instead of silent `None`s.

## Environment variables (12-factor)

Machine- or run-specific values are read from the environment with sensible
defaults: `PITGPT_BASE_MODEL` overrides the base model; `HF_HUB_OFFLINE=1` forces
offline mode once the model is cached. `.env.example` documents them. Secrets and
machine paths stay out of the code.

## Logging, not printing

Library modules use the `logging` module (`logger = logging.getLogger(__name__)`)
so callers control verbosity and destination. Only the thin CLI entry points and
final results use `print`. This is the standard split: libraries log, apps print.

## Error handling

Failures are surfaced with actionable messages rather than swallowed. Examples:
`data_module.load_clean_rows` raises a `FileNotFoundError` that *tells you which
command to run*; `model_service` degrades gracefully to the base model when no
adapter is present and logs a warning; `validate_dataset` cleanly separates
malformed rows instead of crashing on the first bad one.

## Type hints & style

Public functions carry type hints (`-> str`, `list[dict[str, str]]`, …) for
editor support and self-documentation. Code targets PEP 8 with a 100-char line
length (configured for `black`/`ruff` in `pyproject.toml`). Descriptive names are
preferred over cleverness — readability first.

## Tests

`tests/` covers the pure-Python logic that doesn't need a GPU: dataset build
(size, determinism, uniqueness), the validator (malformed/duplicate handling),
config invariants (effective batch = 16, LoRA targets), the 20-prompt eval set,
and that every module imports without side effects (no model load at import
time). Run with `pytest`. GPU-dependent paths are verified manually and via the
baseline/evaluation scripts.

## Thin CLI scripts

`scripts/` holds one-line entry points (`build_dataset.py`, `run_baseline.py`,
`train.py`, `evaluate.py`, `chat.py`, `check_gpu.py`) that just call into the
package. This keeps "how to run it" discoverable without burying logic in scripts.

## The result

A reader can open the repo, read `README.md` for the map, `docs/` for the *why*,
`config/settings.py` for every knob, and run any phase with one command. That is
the difference between a notebook dump and a project.
