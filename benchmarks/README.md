# Benchmarks

Reproduces the benchmark matrix (method comparison + parallelization study) on
the bundled multi-objective multi-dimensional knapsack (MOMKP) instances.

Requires a solver. Install with `uv sync --extra highs` before running.

## Quick start

```bash
uv run python -m benchmarks --profile quick
```

Writes JSON to `benchmarks/results/latest.json` and prints a summary table.

## Profiles

| Profile         | Cases | Scenarios                                            | Repeats |
| --------------- | ----- | ---------------------------------------------------- | ------- |
| `quick`         | `2kp50` | `augmecon_r`, `parallel_default`                   | 1       |
| `paper`         | all 7 | `augmecon`, `augmecon_2`, `augmecon_r`, `parallel_default` | 3 |
| `full`          | all 7 | all 8 scenarios                                      | 1       |

## Cases

| Case    | Objectives | Grid points | Nadir              |
| ------- | ---------- | ----------- | ------------------ |
| 2kp50   | 2          | 492         | -                  |
| 2kp100  | 2          | 823         | -                  |
| 2kp250  | 2          | 2534        | -                  |
| 3kp40   | 3          | 540         | 1031, 1069         |
| 3kp50   | 3          | 847         | 1124, 1041         |
| 4kp40   | 4          | 141         | 138, 106, 121      |
| 4kp50   | 4          | 53          | 718, 717, 705      |

## Scenarios

| Scenario                  | Workers | flag  | bypass | early\_exit | work\_distribution | flag\_policy |
| ------------------------- | ------- | ----- | ------ | ----------- | ------------------ | ------------ |
| `augmecon`                | 1       | off   | off    | off         | fixed              | local        |
| `augmecon_2`              | 1       | off   | on     | off         | fixed              | local        |
| `augmecon_r`              | 1       | on    | on     | on          | fixed              | local        |
| `parallel_default`        | N       | on    | on     | on          | dynamic            | shared       |
| `parallel_simple`         | N       | on    | on     | on          | fixed              | local        |
| `parallel_no_redivide`    | N       | on    | on     | on          | fixed              | shared       |
| `parallel_no_shared_flag` | N       | on    | on     | on          | dynamic            | local        |
| `parallel_outer_grid`     | N       | on    | on     | on          | outer\_grid        | shared       |

## Cores sweep

```bash
uv run python -m benchmarks --cores-sweep --cases 3kp40,3kp50,4kp40,4kp50
```

Sweeps `--workers` in `range(2, 49, 2)` on the `parallel_default` scenario.
Two-objective cases are skipped (the parallel inner loop has nothing to split).

## CLI options

```
--engine NAMES           Comma-separated engines or 'all' (pyaugmecon, augmecon-py)
--profile NAME           quick | paper | full (default: quick)
--cases NAMES            Comma-separated case names or 'all'
--scenarios NAMES        Comma-separated scenario names or 'all'
--repeats N              Runs per (engine, case, scenario)
--workers N              Worker count for scenarios that don't pin one
--cores-sweep            Sweep 2..48 cores on parallel_default
--solvers NAMES          Comma-separated solver families (default: highs)
--solver-opt KEY=VALUE   Repeatable solver option
--output PATH            JSON output path
```

## Engines

| Engine        | Description                                              | Parallel |
| ------------- | -------------------------------------------------------- | -------- |
| `pyaugmecon`  | This project (parallel-capable AUGMECON-R).              | yes      |
| `augmecon-py` | Reference single-process implementation (`../augmecon-py`). | no    |

The `augmecon-py` engine builds the same knapsack instance from the
bundled `pyaugmecon.data` so both engines solve byte-identical problems. It is
loaded lazily and skipped if the sibling repo is missing. Parallel scenarios
are skipped for `augmecon-py`.

## Output

Each run records runtime, Pareto point count, hypervolume, models solved, and
whether results match the baseline (the first scenario for each (engine, case)
pair). The summary table shows median runtime and speedup vs that baseline.

## Adding a case

Edit `cases.py`:

```python
BENCHMARK_CASES["new_case"] = BenchmarkCase(
    name="new_case",
    objective_count=3,
    grid_points=200,
    nadir_points=[10, 20],
)
```

Place fixture CSVs at `src/pyaugmecon/data/<new_case>/`.
