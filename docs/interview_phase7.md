# Phase 7 — Interview Questions & Model Answers

### Q1. Why use a `src/` layout instead of putting the package at the repo root?

**Answer.** The `src/` layout forces you to install the package to import it,
which means your tests and app import the *installed* code, not a stray copy that
happens to be on `sys.path` because it's the current directory. That catches
packaging bugs early (missing modules, bad `__init__`) and makes "works locally,
breaks when installed" failures nearly impossible. It also keeps the import
namespace clean and mirrors how the code will actually be distributed.

### Q2. Why frozen dataclasses for config instead of a dict or module-level globals?

**Answer.** A dict returns `None` (or `KeyError`) for a typo'd key at runtime,
often far from the mistake; module globals are easy to mutate accidentally.
Frozen dataclasses give attribute access with IDE autocomplete and type hints, are
immutable (so config can't drift mid-run), and raise `AttributeError` immediately
on a typo. It's a single, discoverable source of truth that fails loudly.

### Q3. When should code `print` vs `log`?

**Answer.** Libraries should `log` and applications should `print`. Logging lets
the *caller* decide level, format, and destination, and can be silenced or routed
without editing the library. `print` is for the final user-facing output of a CLI
or the app itself. Mixing them — libraries printing — means you can't control
verbosity and you pollute stdout that a caller might be parsing.

### Q4. What does "graceful degradation" look like in this project?

**Answer.** The clearest example is the app: if the trained adapter isn't on disk,
`model_service` loads the base model instead of crashing, logs a warning, and the
UI header says "base model (no adapter)". Similarly the model loader falls back to
fp32 on a CPU-only machine so tests can run without a GPU, and the dataset loader
raises a message that names the exact command to fix the problem. The system stays
usable and honest when a dependency is missing.

### Q5. How do you test ML code that depends on a GPU and a large model?

**Answer.** Separate the logic that *doesn't* need them and test that thoroughly —
dataset building/validation, config invariants, prompt formatting, and that
modules import without side effects (crucially, no model download at import time).
For the GPU path, keep it thin and deterministic where possible, provide a CPU
fallback so it at least runs in CI, and verify end-to-end behaviour with scripted
runs (baseline capture, evaluation report) whose outputs are committed and can be
diffed. You don't unit-test the GPU; you smoke-test the pipeline and unit-test the
logic around it.

### Q6. What makes this repo "GitHub-ready" versus a pile of scripts?

**Answer.** A reader can orient in minutes: a professional README with an
architecture diagram and one-command instructions per phase; a single config
source of truth; a modular `src/` package with clear boundaries; type hints,
logging, and error messages; a test suite; a license; a `.gitignore` that keeps
weights and caches out; and `docs/` explaining the *why* behind each decision.
Nothing requires tribal knowledge to run or understand.
