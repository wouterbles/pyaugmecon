# Benchmarks

Benchmark setup lives entirely in YAML. Each `runs:` entry is
self-contained — no defaults, no inheritance, no CLI fallbacks.

## Run

```bash
uv run python -m benchmarks
```

or with a specific plan:

```bash
uv run python -m benchmarks --plan benchmarks/plans/method_comparison.yaml
uv run python -m benchmarks --plan benchmarks/plans/parallel_scaling.yaml --output benchmarks/results/scaling.json
```

Writes JSON to `benchmarks/results/latest.json`. The runner is resumable
by default: it reads existing runs from `--output` and skips jobs already
completed.

## Plan format

```yaml
include:
  - _cases.yaml
  - _scenarios.yaml

runs:
  - name: example
    engine: pyaugmecon       # one of the engines in _scenarios.yaml's registry
    solver: gurobi           # solver name (string or list of strings)
    case: [3kp40, 3kp50]     # case name(s), or "all"
    scenario: [augmecon_r]   # scenario name(s), or "all"
    workers: [1, 14]         # worker count(s) — list = sweep
    samples: 3               # repeated samples per setup
    solver_options: {}       # optional; default {}
```

Required per run: `engine`, `solver`, `case`, `scenario`, `workers`,
`samples`. Only `solver_options` is optional.

`include:` pulls `cases` and `scenarios` from sibling YAML files. The
plan's own `cases:` / `scenarios:` blocks (if any) override included
entries with the same name.

## Plans included

- `default.yaml` — AUGMECON-R serial and parallel on `2kp50` under HiGHS.
  Quick check that the runner works end-to-end.
- `method_comparison.yaml` — AUGMECON, AUGMECON-2, AUGMECON-R, and
  parallel AUGMECON-R on every case. Captures CPU time, models solved,
  infeasibilities, dominated solutions, Pareto-front size, and HV
  indicator per method. Gurobi, 3 samples.
- `engine_comparison.yaml` — PyAUGMECON v2 (serial + parallel) vs
  AUGMECON-Py (serial only) on every case under Gurobi. Solver is held
  constant so the engine axis is the only variable.
- `solver_comparison.yaml` — PyAUGMECON v2 with both Gurobi and HiGHS,
  in serial and parallel modes. Shows whether the open-source solver
  scales similarly to the commercial one.
- `parallel_scaling.yaml` — worker-count sweep (1..28) for parallel
  AUGMECON-R on the 3- and 4-objective cases. The `workers=1` point
  doubles as the serial baseline. Two-objective cases skipped (no outer
  grid loop). Gurobi, 1 sample.
- `density_sweep.yaml` — AUGMECON-R on the `2kp100` dataset at four
  grid densities (10, 50, 200, 823). Each density is its own case via
  `base_case`, so each output row is self-contained.

### Per-case data aliases

A case may set `base_case: <name>` to use a different bundled dataset
than its own name. Used by `density_sweep.yaml`.

## Output JSON

Each run record includes: `engine`, `solver`, `solver_options`, `case`,
`scenario`, `workers`, `sample`, `runtime_seconds`, `pareto_points`,
`dominated_points`, `models_solved`, `models_infeasible`, `hv_indicator`,
`grid_points`, `objective_count`. The top-level `environment` block
records platform, Python, and key package versions for reproducibility.

The `summary` block aggregates across samples per
`(engine, solver, case, scenario, workers)` group with
`mean_s`/`median_s`/`min_s`/`max_s`, `speedup` (vs single-worker median for
the same engine/solver/case), `parity` (cross-sample agreement on the
rounded Pareto + payoff snapshot), and `xparity` (parity vs PyAugmecon
for the same solver/case).

## CLI options

```
--plan PATH      Path to benchmark YAML plan
--output PATH    JSON output path
--[no-]resume    Resume from existing output (default: on)
```
