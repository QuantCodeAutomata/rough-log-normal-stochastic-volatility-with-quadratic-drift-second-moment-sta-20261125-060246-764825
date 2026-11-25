# Repository guide for coding agents

This repository reproduces selected numerical results from Sepp, A. and
Rakhmonov, A., *"Rough log-normal stochastic volatility with quadratic
drift: second-moment stability and the martingale property under positive
correlation"*: the second-moment stability analysis of the linear-drift
log-normal Volterra model (Table 2, Figure 2), the Markovian-lift
operator-splitting simulation scheme that underlies the Monte Carlo
experiments (discrete martingale check), the martingale-property test under
positive correlation (Table 3, Figure 3(a)), the time-averaged second-moment
bound under quadratic drift (Figure 3(b)), and a self-test of the
martingale-constrained smile-calibration machinery against the paper's own
published Table 4 parameter vectors (the paper's Bitcoin/Ether option-quote
data could not be obtained; see `results/RESULTS.md`).

## Layout

```
src/            flat library modules: fractional-kernel resolvent and
                 Mittag-Leffler function (kernels.py, mittag_leffler.py),
                 N-factor Markovian lift of the kernel (markovian_kernel.py),
                 symmetric operator-splitting Monte Carlo engine
                 (simulation.py), linear-drift exact moments and the
                 Proposition 3.7 bound (bounds.py), option pricing under both
                 kernels (pricing.py), smile calibration (calibration.py),
                 and all parameters (config.py)
exp/            one script per experiment, run as `python -m exp.exp_N`
tests/          pytest unit tests of the src/ components
results/        aggregate metrics (.json/.csv) and static .png figures;
                 overwritten by each experiment run
data/           data-preparation package (empty: no experiment here loads
                 external market data)
```

No market data is used: experiments 1-4 are closed-form/simulation studies
with parameters taken from the paper, and experiment 5 could not obtain the
paper's Deribit option-quote data (not served by the QuantX Dataloader or
Massive REST APIs, and available only on request from the paper's
corresponding author), so it only self-tests the calibration code against
the paper's own published parameter vectors. No API keys are required.

## Conventions

- All experiment parameters (Hurst exponents, kappa_1/kappa_2 grids, kernel
  nodes and weights, Monte Carlo path counts, grid resolutions, random
  seeds) live only in `src/config.py` as module-level `UPPER_CASE`
  variables, each with a comment naming its meaning, unit, and the paper
  section or equation it comes from (or "not specified in the paper").
  Do not hardcode parameters inside `src/` modules or `exp/` scripts.
- Library code is flat under `src/`; experiment scripts only call `src/`
  functions by their documented signatures and never redefine a private
  copy of a shared component (kernel, simulation engine, or pricer).
- Data preparation code, if any is ever added, goes only in `data/`
  (`data_loader.py` for market data, `generate_synthetic.py` for synthetic
  data); none is present today because no experiment needs it.
- Tests run with `python -m pytest -m "not market_data"` from the
  repository root; there are currently no `market_data`-marked tests
  because no experiment loads market data, but the flag is kept for
  consistency with the shared testing convention.
- Experiments run from the repository root as `python -m exp.exp_N`; do
  not edit `sys.path` in code.
- Keep API keys only in an uncommitted `.env` file; none are needed by the
  code in this repository today.

## Rerunning

Each `python -m exp.exp_N` run overwrites its files in `results/`
(`expN_metrics.json` and any `.png`/`.csv` figures). After a rerun, check
the new files and update the numbers, tables and verdicts in
`results/RESULTS.md` by hand; they are not regenerated automatically.
`reproduce.sh` runs the test suite followed by all five experiments in
order.
