# Contributing

## Dev setup

```bash
git clone https://github.com/wouterbles/pyaugmecon.git
cd pyaugmecon
uv sync
```

`uv sync` installs the package in editable mode with all dev dependencies and HiGHS.

## Before submitting

```bash
uv run ruff format --check .   # formatting
uv run ruff check .            # lint
uv run ty check src/pyaugmecon # type check
uv run pytest                  # fast tests (~2s)
uv run pytest -m ""            # full suite (slower, needs solver)
```

Pre-commit hooks (optional):

```bash
prek install
prek run --all-files
```

## Tests

- `uv run pytest` : defaults to quick tests (skips `slow` and `knapsack` markers).
- `uv run pytest -m ""` : everything.
- `uv run pytest -m knapsack` : knapsack regression only.
- `uv run pytest -m slow` : slow tests only.

## Pull requests

- Keep PRs focused and small.
- Match the existing code style (Ruff handles most of it).
- Update the changelog under `## Unreleased` if the change is user-facing.
