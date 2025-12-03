# Repository Guidelines

## Project Structure & Module Organization
- Core Python source lives in `nanochat/` (e.g., `gpt.py`, `engine.py`, `dataset.py`, `tokenizer.py`) and drives training, inference, and evaluation. Frontend lives in `nanochat/ui.html`.
- Training/eval entrypoints are in `scripts/` (e.g., `base_train.py`, `mid_train.py`, `chat_web.py`). Reusable tasks are under `tasks/`.
- Rust tokenizer trainer resides in `rustbpe/` (built via maturin/pyo3). Tests are in `tests/`, and helper assets/examples are in `dev/`.
- End-to-end run scripts are `speedrun.sh` (budget d20) and `run1000.sh` (d32); they orchestrate the full pipeline on multi-GPU boxes.

## Build, Test, and Development Commands
- Install deps (preferred): `uv sync --all-extras --dev` inside an activated virtualenv; alternatively `pip install -e .[dev]`.
- Build the Rust tokenizer locally when iterating on it: `maturin develop` from repo root (respects `rustbpe/Cargo.toml`).
- Fast CPU/MPS-sized smoke: `bash dev/runcpu.sh`. Full-budget training: `bash speedrun.sh` or `bash run1000.sh` on 8×H100.
- Run the web chat UI (after training): `python -m scripts.chat_web` and open the printed host:port.
- Run tests: `python -m pytest` for all, or `python -m pytest tests/test_rustbpe.py -v -s` for tokenizer checks.

## Coding Style & Naming Conventions
- Follow PEP8 with 4-space indents; use `snake_case` for functions/variables and `CamelCase` for classes. Keep modules minimal and readable—avoid unnecessary abstractions.
- Favor type hints and short, purposeful docstrings for new public functions; keep configuration centralized through `configurator.py` where possible.
- Logging/prints should be sparse and actionable; prefer deterministic seeds when adding training/eval code paths.

## Testing Guidelines
- Use `pytest`; slow workloads can be marked with `@pytest.mark.slow` (see `pyproject.toml`). Name tests `test_*.py` and mirror module names.
- Add targeted unit tests when touching tokenization, checkpointing, or distributed engine changes; prefer CPU-sized fixtures and minimal data slices to keep runtime low.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit messages (e.g., `Add KV cache masking guard`). Keep commits focused.
- In PR descriptions, summarize behavior changes, list tested commands, and link related issues/discussions. Include sample logs/metrics for training-affecting changes.
- Per repo policy, disclose any LLM-generated code you did not fully author or understand. Document new flags/config fields in the relevant script docstrings and README sections if users must update workflows.
