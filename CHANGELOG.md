# Changelog

## 2.0.1 - 2026-04-27

- Removed pre-v2 dead code (model.py, pyaugmecon.py, process_handler.py, queue_handler.py, solver_adapter.py, solver_process.py) superseded by the solver/ subpackage.


## 2.0.0 - 2026-04-26

### Breaking changes

- Moved the package to `src/pyaugmecon` and removed the options module. Use `PyAugmeconConfig` directly.
- Replaced the broad options object with one strict Pydantic config model. Unsupported setting names are rejected instead of being silently accepted.
- Kept the public result surface small: removed undocumented `PyAugmecon` attributes such as `coverage_*`, `exact_guarantee`, `objective_order_*`, `nadir_source`, `grid_points_effective`, and `summary`.
- Exposes advanced parallel controls as `work_distribution` and `flag_policy`.

### Core behavior

- Added explicit `exact` and `sampled` run modes.
- Built epsilon levels deterministically and added objective ordering with `objective_order="auto_range"` or `"given"`.
- Automatically deactivates all user objectives before solving.
- Checks for reserved internal component names before modifying the Pyomo model.
- Computes auto-safe nadir bounds when explicit `nadir_points` are not provided.
- Keeps decision-variable payloads optional with `store_decision_variables`.
- Normalizes objective keys deterministically with `round_decimals`. Workers also snap near-integer noise on objective values via `objective_tolerance` before sending results.

### Parallel solve path

- Uses spawn-based multiprocessing with shared-memory model handoff.
- Splits work with `work_distribution="dynamic"`, `"fixed"`, or `"outer_grid"`. The default `"auto"` picks `"outer_grid"` for exact multi-worker runs and `"dynamic"` otherwise.
- Uses `flag_policy="local"` or `"shared"`. The default `"auto"` picks `"shared"` for exact multi-worker runs and `"local"` otherwise.
- Adds clearer worker failure, timeout, early-stop, job queue, result queue, and error queue handling.
- Workers emit structured `Solution` records instead of nested result maps.
- The worker hot loop caches Pyomo component handles and objective expressions once per worker to reduce repeated Pyomo lookups.
- Shared outer-grid skip state avoids repeat infeasible solves between workers; missed concurrent updates can only cost an extra solve, not a wrong answer.

### Solver support

- Sets HiGHS as the default solver family.
- Adds fallback chains for HiGHS, Gurobi, CPLEX, XPRESS, CBC, and SCIP families.
- Adds `solver_options`, `solver_io`, and `solve_warmstart` controls.
- Handles solver statuses through a shared adapter so infeasible, infeasible-or-unbounded, time-limit, and unsupported terminations are reported consistently.

### Logging and artifacts

- Replaced print output with `tqdm` progress and Loguru-backed run logging.
- Added `log_sink=` to forward run messages to objects with `.info(message)`.
- Added `artifact_folder`, `artifact_name`, `write_csv`, `progress_bar`, `log_to_console`, and `process_logging`.
- Writes payoff, grid, and solution tables as CSV files through a small tabular helper.

### Docs, examples, and packaging

- Solver backends are opt-in through extras: `pyaugmecon[highs]`, `pyaugmecon[gurobi]`, `pyaugmecon[xpress]`, `pyaugmecon[cbc]`. `cbc` now installs `cbcbox` to match Pyomo's executable backend path. `scip` remains supported through Pyomo when a `scip` executable is installed. The base install does not pull in any solver.
- Documented solver fallback order, backend selection rationale, and common free/community license notes.
- Packaged the knapsack CSV data used by `pyaugmecon.example_models`, so the bundled examples work from installed wheels.
- Rewrote benchmarks as a modular CLI (`benchmarks/cli.py`, `cases.py`, `engines.py`) with engine pluggability supporting both this project and the `augmecon-py` implementation. Bundled knapsack benchmark cases are maintained and documented in `benchmarks/README.md`.
- Restructured tests into `tests/unit`, `tests/integration`, and `tests/support`, with knapsack regression reference data under `tests/input/`.
- Switched project tooling to `uv`, Ruff, ty, pytest, and `prek`.
- Replaced the previous CI setup with GitHub Actions for Ruff, ty, and pytest on Python 3.12, 3.13, and 3.14.
- Added release workflow to verify builds and publish to PyPI.
- Removed obsolete files: `dev/make_distribution.sh`, `gurobi.env`, and `pytest.ini`.

### Fixed Issues and PRs

- Queue and scheduling work: [Issue #14](https://github.com/wouterbles/pyaugmecon/issues/14).
- Auto publish pipeline: [Issue #15](https://github.com/wouterbles/pyaugmecon/issues/15).
- CI coverage updates: [Issue #16](https://github.com/wouterbles/pyaugmecon/issues/16).
- Solution storage updates: [Issue #17](https://github.com/wouterbles/pyaugmecon/issues/17).
- Solver support including CBC: [Issue #20](https://github.com/wouterbles/pyaugmecon/issues/20).
- Objective auto-deactivation: [Issue #22](https://github.com/wouterbles/pyaugmecon/issues/22).
- Reserved-name conflict checks: [Issue #23](https://github.com/wouterbles/pyaugmecon/issues/23).
- Thanks [@yvanoers](https://github.com/yvanoers) for [PR #6](https://github.com/wouterbles/pyaugmecon/pull/6) and [PR #7](https://github.com/wouterbles/pyaugmecon/pull/7).
- Thanks [@kschulze26](https://github.com/kschulze26) for [PR #25](https://github.com/wouterbles/pyaugmecon/pull/25).

## 1.0.8 - 2024-02-26

- Add method to check if user-provided model component names mask pyaugmecon-added component names
- Update LICENSE

## 1.0.7 - 2024-01-29

- Update readme to include other solvers
- Fix Pandas warning

## 1.0.6 - 2024-01-29

- Fix dependencies

## 1.0.5 - 2024-01-29

- Update dependencies

## 1.0.4 - 2023-08-14

- Correctly enqueue items such that the work can be redistributed when a process finishes its work

## 1.0.3 - 2023-08-11

- Use new Pyomo 6.6.1 method to `close()` Gurobi and prevent duplicate instances, thanks @torressa

## 1.0.2 - 2023-08-10

- Update dependencies

## 1.0.1 - 2023-07-21

- Make QueueHandler work on MacOS #5, thanks @yvanoers

## 1.0.0 - 2023-04-17

- Add new methods for solution retrieval

## 0.2.1 - 2023-04-04

- Add Python 3.10 to supported versions

## 0.2.0 - 2023-04-04

- Decision variables are now stored in a dictionary together with the objective values
- More details in the [documentation](https://github.com/wouterbles/pyaugmecon#pyaugmecon-solutions-details)

## 0.1.9 - 2023-01-09

- Again fix package dependencies to prevent breaking changes
- Update dependencies

## 0.1.8 - 2022-12-15

- Relax dependency requirements

## 0.1.7 - 2021-12-10

- Remove default solver options when setting them to `None`

## 0.1.6 - 2021-12-08

- Fix issue with mixed min/max objectives

## 0.1.5 - 2021-10-07

- Fix dependency versions after issue with change in Pymoo API

## 0.1.4 - 2021-06-21

- Incorrectly bumped version, no change from 0.1.3

## 0.1.3 - 2021-06-19

- Trigger new release for Zenodo

## 0.1.2 - 2021-05-30

- Move process timeout check to seperate thread to prevent a deadlock

## 0.1.1 - 2021-05-29

- Add more detailed installation instructions to README and fix typos

- Add CHANGELOG

## 0.1.0 - 2021-05-29

- 🎉 First published version!

- Alpha quality
