<div align="center">
<img src="https://raw.githubusercontent.com/wouterbles/pyaugmecon/main/logo.png" alt="Logo" width="330">
</div>

## Multi-objective optimization for Pyomo using the AUGMECON method

[![Tests](https://github.com/wouterbles/pyaugmecon/actions/workflows/ci.yml/badge.svg)](https://github.com/wouterbles/pyaugmecon/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyaugmecon)](https://pypi.org/project/pyaugmecon)
[![Python](https://img.shields.io/pypi/pyversions/pyaugmecon)](https://pypi.org/project/pyaugmecon)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://github.com/wouterbles/pyaugmecon/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/pyaugmecon)](https://pepy.tech/project/pyaugmecon)
[![Lint: Ruff](https://img.shields.io/badge/lint-ruff-46a2f1.svg)](https://github.com/astral-sh/ruff)
[![Type check: ty](https://img.shields.io/badge/type%20check-ty-1f6feb.svg)](https://docs.astral.sh/ty/)
[![DOI](https://zenodo.org/badge/336300468.svg)](https://zenodo.org/badge/latestdoi/336300468)

PyAUGMECON solves multi-objective optimization problems defined in [Pyomo](https://pyomo.readthedocs.io/) using the augmented epsilon-constraint method (AUGMECON) and its variants.

- Generates the Pareto front by solving a sequence of single-objective subproblems with epsilon constraints.
- Supports exact mode (full grid coverage) and sampled mode (user-defined discretization).
- Implements AUGMECON early exit, AUGMECON2 bypass, and AUGMECON-R pruning to skip redundant solves.
- Runs subproblems in parallel across multiple processes with spawn-safe serialization.
- Resolves solver backends automatically across HiGHS, Gurobi, CPLEX, XPRESS, CBC, and SCIP families.

[GAMS implementations](#references) of the method were provided by the original authors. To the best of our knowledge, this is the first publicly available Python implementation.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [API](#api)
- [Benchmarks](#benchmarks)
- [Tests](#tests)
- [Notes](#notes)
- [Development](#development)
- [References](#references)
- [Changelog](#changelog)
- [Citing](#citing)

## Installation

PyAUGMECON does not bundle a solver. Pick one when installing:

```bash
# Recommended: uv (https://docs.astral.sh/uv/)
uv add "pyaugmecon[highs]"

# Or with pip
pip install "pyaugmecon[highs]"
```

Available solver extras:

| Extra | Solver |
| --- | --- |
| `pyaugmecon[highs]` | HiGHS (open source, no license required). |
| `pyaugmecon[gurobi]` | Gurobi (requires license). |
| `pyaugmecon[xpress]` | XPRESS (uses the license available to the installed `xpress` package). |
| `pyaugmecon[cbc]` | CBC executable via `cbcbox` (matches Pyomo `cbc` backends). |

Solvers not bundled as extras but supported when their binary is on `PATH`:

| Solver | Link | Solver name |
| --- | --- | --- |
| SCIP | [scipopt.org](https://www.scipopt.org) | `scip` |
| CPLEX | [ibm.com/products/ilog-cplex-optimization-studio](https://www.ibm.com/products/ilog-cplex-optimization-studio) | `cplex` |

Once installed, pass `solver_name="scip"` or `solver_name="cplex"`. You can also install CBC manually instead of `pyaugmecon[cbc]`; `cbcbox` is typically faster than a system-installed CBC binary (e.g. 85 s vs 471 s on `3kp40`), as it ships AVX2-optimized builds with bundled OpenBLAS.

## Quick start

Define a Pyomo model with objectives in an `ObjectiveList` named `obj_list`, then pass it to PyAUGMECON. Defaults are sensible, so a minimal run takes no config:

```python
from pyomo.environ import ConcreteModel, Constraint, ObjectiveList, Var, NonNegativeReals, maximize
from pyaugmecon import PyAugmecon, PyAugmeconConfig


def two_objective_model():
    model = ConcreteModel()
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    model.con1 = Constraint(expr=model.x1 <= 20)
    model.con2 = Constraint(expr=model.x2 <= 40)
    model.con3 = Constraint(expr=5 * model.x1 + 4 * model.x2 <= 200)
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=model.x1, sense=maximize)
    model.obj_list.add(expr=3 * model.x1 + 4 * model.x2, sense=maximize)
    return model


if __name__ == "__main__":
    solver = PyAugmecon(two_objective_model(), PyAugmeconConfig())
    result = solver.solve()

    print(result.points)
    print(result.payoff_table)
```

The `if __name__ == "__main__"` guard is intentional. PyAUGMECON starts worker processes with Python's `spawn` mode, which imports your script in each child process. The guard keeps child processes from running your top-level solve code again.

> [!IMPORTANT]
> Always wrap your `PyAugmecon(...)` / `.solve()` calls in `if __name__ == "__main__"` when using `workers > 1`, otherwise child processes will re-execute your script on import.

A more configured example using a bundled knapsack model, showing options that are available but optional in the minimal example above:

```python
from pyaugmecon import PyAugmecon, PyAugmeconConfig
from pyaugmecon.example_models import kp_model

if __name__ == "__main__":
    config = PyAugmeconConfig(
        name="3kp40",
        mode="exact",
        solver_name="highs",
        nadir_points=[1031, 1069],
        store_decision_variables=True,
        workers=4,
    )
    solver = PyAugmecon(kp_model("3kp40", 3), config)
    result = solver.solve()

    decision_vars = result.variables_for(result.points[0])
```

## API

### Constructor

`PyAugmecon(model, config, *, log_sink=None)` where:

- `model` : unsolved Pyomo `ConcreteModel` with an `ObjectiveList` named `obj_list`.
- `config` : a `PyAugmeconConfig` instance or a plain dict with the same fields.
- `log_sink` : optional object with `.info(message)` method for forwarding PyAUGMECON log messages to your own logging system.

### Methods

| Method | Returns |
| --- | --- |
| `solve()` | Runs the algorithm and returns `PyAugmeconResult`. |

### Result object

| Attribute | Description |
| --- | --- |
| `solutions` | Final Pareto solutions as `Solution` records. |
| `points` | Objective vectors from `solutions`. |
| `count` | Number of Pareto points. |
| `total_points` | Number of distinct objective vectors (after rounding) before Pareto filtering. |
| `runtime_seconds` | Wall-clock solve time. |
| `payoff_table` | Lexicographic payoff table. |
| `models_solved` | Number of subproblems solved. |
| `models_infeasible` | Number of infeasible subproblems. |
| `visited_points` | Number of grid points visited by workers. |
| `grid_point_count` | Planned grid point count. |
| `hypervolume()` | Hypervolume of the Pareto front. Computed lazily on first call. |

Each `Solution` record has:

- `point`: the objective vector.
- `variables`: variable values for that point, or `None` when decision storage is off.

Example `result.solutions` layout with `store_decision_variables=True`:

```python
[
    Solution(point=(3, 5), variables={"x": {0: 1, 1: 2}}),
    Solution(point=(4, 4), variables={"x": {0: 3, 1: 4}}),
]
```

### Config

`PyAugmeconConfig` is a Pydantic model. Pass it directly or as a plain dict.

| Field | Default | Description |
| --- | --- | --- |
| `name` | `"{solver_name}-{mode}"` (auto) | Run name for logs and artifacts. |
| `mode` | `"exact"` | `"exact"` (full grid) or `"sampled"` (user-defined discretization). |
| `sample_points` | `None` | Grid density per constrained objective. Required in sampled mode. |
| `nadir_points` | `None` | Explicit nadir values for constrained objectives. Auto-computed if omitted. |
| `nadir_strategy` | `"safe"` | How nadirs are auto-computed when `nadir_points` is omitted: `"safe"` or `"payoff"`. |
| `nadir_undercut` | `0.8` | Widening factor used by `nadir_strategy="payoff"`. Smaller values widen the grid more. |
| `objective_order` | `"auto_range"` | `"auto_range"` (sort by range, descending) or `"given"`. |
| `workers` | `cpu_count()` | Number of worker processes. |
| `work_distribution` | `"auto"` | How grid points are assigned to workers: `"auto"`, `"dynamic"`, `"fixed"`, or `"outer_grid"`. |
| `flag_policy` | `"auto"` | Whether AUGMECON-R flag information is private to each worker (`"local"`) or shared between workers (`"shared"`). |
| `process_timeout` | `None` | Timeout in seconds for the entire run. |
| `solve_warmstart` | `True` | Pass previous solution to solver when supported. |
| `store_decision_variables` | `False` | Keep variable values with each Pareto point. |
| `early_exit` | `True` | AUGMECON early exit: stop iterating when infeasible. |
| `bypass` | `True` | AUGMECON2 bypass: skip non-binding constraint levels. |
| `flag` | `True` | AUGMECON-R pruning: mark visited grid points to avoid re-solving. |
| `penalty_weight` | `1e-3` | Slack penalty multiplier in the augmented objective. |
| `objective_tolerance` | `1e-6` | Tolerance for deduplicating objective values. |
| `round_decimals` | `9` | Decimal precision for solution keys. |
| `solver_name` | `"highs"` | Solver family or explicit Pyomo backend name. |
| `solver_io` | `None` | Optional Pyomo backend hint. |
| `solver_options` | `{}` | Options passed to the solver backend. |
| `write_csv` | `True` | Write payoff, grid, and final solution tables as CSV files. |
| `artifact_folder` | `"logs"` | Output directory for logs and CSV artifacts. |
| `artifact_name` | auto | Explicit artifact/log basename. Defaults to `<name>_<timestamp>`. |
| `progress_bar` | `True` | Show progress output. |
| `log_to_console` | `True` | Emit logs to stdout/stderr. |
| `process_logging` | `False` | Enable per-worker logging. Reduces performance. |

### Advanced work and pruning settings

Most users only need `workers`. Leave `work_distribution="auto"` and `flag_policy="auto"` unless you are benchmarking, comparing algorithm variants, or debugging parallel performance.

`work_distribution` controls how the flat epsilon grid is split:

- `"auto"` selects `"outer_grid"` for exact multi-worker runs and `"dynamic"` otherwise.
- `"dynamic"` uses one shared queue. A worker takes the next small range as soon as it finishes its current range. This is simple and handles uneven solve times well.
- `"fixed"` gives each worker one continuous part of the grid. There is no work stealing, so it is useful for controlled benchmarks but can be slower when some points take longer to solve.
- `"outer_grid"` groups points by the slower-changing constrained objectives: a worker gets blocks where the outer objective levels stay together while the innermost objective level changes fastest. This helps shared pruning because one infeasible solve can tell other workers to skip later inner levels for the same outer objective levels.

`flag_policy` controls who can see AUGMECON-R flag information:

- `"auto"` selects `"shared"` for exact multi-worker runs and `"local"` otherwise.
- `"local"` keeps skip information inside each worker. It has the least coordination overhead.
- `"shared"` stores flag information in shared memory. Workers can avoid repeating solves that another worker already proved redundant. This is most useful with `work_distribution="outer_grid"`.

The skip controls are still separate because they are useful for benchmarks and research:

- `early_exit=True` stops scanning the innermost objective levels after an infeasible subproblem because later inner levels are at least as hard.
- `bypass=True` uses positive slack to skip nearby epsilon levels that lead to the same objective vector.
- `flag=True` records skip counts so later grid points can be skipped before calling the solver.

### Nadir computation

The lower bound of the epsilon grid for each constrained objective (its "nadir") sets how wide the search has to be. You can supply explicit values via `nadir_points`; otherwise PyAUGMECON computes them with one of two strategies:

- `nadir_strategy="safe"` (default) solves one extra single-objective minimization per constrained objective over the entire feasible region. The result is a provably safe lower bound on the true nadir, so no Pareto-optimal point can fall below the grid. It costs `n - 1` extra setup solves and can produce a wider grid than strictly needed; AUGMECON-R bypass usually skips the slack cells without solver calls.
- `nadir_strategy="payoff"` reuses the lex payoff table column minima and widens them by `nadir_undercut`. This is the classic AUGMECON / AUGMECON-R convention. It adds no setup solves and yields a tighter grid, but the column minimum is the anti-ideal point, which can be larger than the true nadir for three or more objectives. The `nadir_undercut` factor (default `0.8`) is a safety margin that pushes the bound down; lower values widen the grid, `1.0` disables widening.

Use `"safe"` when correctness must be guaranteed without tuning. Use `"payoff"` to reproduce literature results or when the extra setup solves are expensive (e.g. large MIPs where the relaxation is hard).

### Solver fallback order

When you specify a solver family, PyAUGMECON tries backends in order until one is available:

| Family | Tried in order |
| --- | --- |
| `highs` | `appsi_highs` |
| `gurobi` | `gurobi_direct`, `appsi_gurobi`, `gurobi_persistent`, `gurobi` |
| `cplex` | `appsi_cplex`, `cplex_persistent`, `cplex` |
| `xpress` | `xpress_direct`, `xpress_persistent`, `xpress` |
| `cbc` | `cbc`, `appsi_cbc` |
| `scip` | `scip` |

You can also pass an explicit backend name (e.g. `solver_name="appsi_highs"`).

The order favors in-process Pyomo backends first because PyAUGMECON solves many closely related models and avoids command-line startup cost when possible. HiGHS tries `appsi_highs` first because the `highspy` Python backend (installed via `pyaugmecon[highs]`) is in-process. Commercial solver families prefer direct/APPSI-style Python interfaces, then persistent interfaces, then the broad command-line backend. If you need one exact backend for a benchmark or deployment, pass that backend name directly.

### Solver notes

- `gurobi` and `xpress` extras add their Python bindings, but you still need a working license/setup.
- System-installed CBC may be slower than `cbcbox` due to compiler and BLAS optimizations.
- SCIP and CPLEX need a solver binary installed separately (see installation section above).

Free/community solver options change over time, so check vendor docs before relying on them:

| Solver | Free/community option |
| --- | --- |
| [HiGHS](https://highs.dev/) | Open-source solver, no commercial license needed. |
| [CBC](https://github.com/coin-or/Cbc) | Open-source solver from COIN-OR, no commercial license needed. |
| [SCIP](https://www.scipopt.org/) | Free for academic and non-commercial use. See [SCIP licensing](https://www.scipopt.org/index.php#license). |
| [Gurobi](https://www.gurobi.com/) | Restricted non-production use and academic licenses are available. See [restricted license notes](https://support.gurobi.com/hc/en-us/articles/29682074018833-What-does-Restricted-license-for-non-production-use-only-mean) and [academic licenses](https://www.gurobi.com/academics). |
| [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) | A limited free edition and academic access are available. See [IBM CPLEX pricing FAQ](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing). |
| [XPRESS](https://www.fico.com/en/products/fico-xpress-optimization) | Community licensing is available through the Python package. See [xpress on PyPI](https://pypi.org/project/xpress/). |

Solver parameter references: [HiGHS](https://highs.dev/) | [Gurobi](https://www.gurobi.com/documentation/current/refman/parameters.html) | [CPLEX](https://www.ibm.com/docs/en/icos/22.1.0?topic=tutorials-python-tutorial) | [XPRESS](https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/HTML/chapter6.html) | [CBC](https://coin-or.github.io/Cbc/) | [SCIP](https://www.scipopt.org/doc/html/PARAMETERS.php)

## Benchmarks

See [`benchmarks/README.md`](benchmarks/README.md) for details.

```bash
uv run python -m benchmarks --profile quick
```

## Tests

Most tests require HiGHS (`uv sync --extra highs`).

```bash
uv run pytest                  # default suite (skips slow cases)
uv run pytest -m ""            # full suite
uv run pytest -m knapsack      # knapsack regression only
```

Knapsack datasets are packaged inside `pyaugmecon.data` and used directly by tests and benchmarks.

## Notes

> [!NOTE]
> In exact mode, the grid is derived from the payoff table and covers the full objective range. In sampled mode, `sample_points` controls discretization density.

> [!TIP]
> With multi-process runs, lowering solver-internal thread counts (e.g. `solver_options={"threads": 1}`) can improve total throughput.

- The choice of constrained objectives affects grid traversal order but not the Pareto front.
- Small instances can be slower with `workers > 1` because process startup overhead dominates.
- `flag_policy="shared"` uses inter-process shared memory for pruning state. This adds coordination overhead but can avoid redundant solves across workers.

## Development

```bash
git clone https://github.com/wouterbles/pyaugmecon.git
cd pyaugmecon
uv sync
```

`uv sync` installs the package in editable mode with all dev dependencies. Add `--extra highs` for the solver (required by integration tests and benchmarks):

```bash
uv sync --extra highs
```

Then run any of:

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # format check
uv run ty check src/pyaugmecon # type check
uv run pytest                  # tests
```

### Pre-commit hooks

Hooks are defined in `prek.toml`. Install [prek](https://prek.j178.dev/) once, then enable hooks for this repo:

```bash
prek install
prek run --all-files
```

### Why uv?

[uv](https://docs.astral.sh/uv/) handles Python version management, virtual environments, and dependency resolution in a single tool. It is significantly faster than pip and avoids common pitfalls with environment isolation. The project's `pyproject.toml` and lockfile are designed for uv, but `pip install -e .` works too.

## References

### AUGMECON

- G. Mavrotas, "Effective implementation of the epsilon-constraint method in Multi-Objective Mathematical Programming problems," *Applied Mathematics and Computation*, 213(2), 2009.
- [GAMS implementation](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_epscm.html)

### AUGMECON2

- K. Florios and G. Mavrotas, "An improved version of the augmented epsilon-constraint method (AUGMECON2)," *Applied Mathematics and Computation*, 219(18), 2013.
- [GAMS implementation](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_epscmmip.html)

### AUGMECON-R

- A. Nikas, A. Fountoulakis, A. Forouli, and H. Doukas, *A robust augmented epsilon-constraint method (AUGMECON-R) for finding exact solutions of multi-objective linear programming problems*, 2020.
- [Reference implementation](https://github.com/KatforEpu/Augmecon-R)

### Other

- [Mavrotas PhD thesis](https://www.chemeng.ntua.gr/gm/gmsite_gre/index_files/PHD_mavrotas_text.pdf)
- [Presentation overview](https://www.chemeng.ntua.gr/gm/gmsite_eng/index_files/mavrotas_MCDA64_2006.pdf)

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use PyAUGMECON in academic work, please cite the [Zenodo DOI](https://zenodo.org/badge/latestdoi/336300468).

## Credit

Developed at the Electricity Markets & Power System Optimization Laboratory (EMPSOLab), [Electrical Energy Systems Group](https://www.tue.nl/en/research/research-groups/electrical-energy-systems/), [Department of Electrical Engineering](https://www.tue.nl/en/our-university/departments/electrical-engineering/), [Eindhoven University of Technology](https://www.tue.nl/en/).

Contributors: Wouter Bles, Nikolaos Paterakis
