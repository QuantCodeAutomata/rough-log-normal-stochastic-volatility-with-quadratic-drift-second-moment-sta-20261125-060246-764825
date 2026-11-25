# Rough log-normal stochastic volatility with quadratic drift

Replication of selected numerical results from:

**Sepp, A. and Rakhmonov, A. (2026). "Rough log-normal stochastic volatility
with quadratic drift: second-moment stability and the martingale property
under positive correlation."**

The paper studies a log-normal Volterra stochastic-volatility model in which
the volatility drift is linear-plus-quadratic in the instantaneous
volatility, driven by a fractional (rough) kernel. It derives (i) a
second-moment stability condition for the linear-drift case via the resolvent
of the fractional kernel, (ii) a uniform second-moment bound under quadratic
damping, and (iii) necessary and sufficient conditions on the quadratic
coefficient for the discounted price to be a true martingale under positive
spot/volatility correlation. This repository reproduces the paper's analytic
computations and its Markovian-lifted Monte Carlo simulation scheme, and
checks them against the paper's reported numbers.

No market data is required for experiments 1-4 (closed-form and simulation
studies only). Experiment 5 (smile calibration) would require the paper's
archived Deribit Bitcoin/Ether option quotes, which the paper states are
available only on request from its corresponding author and are not served by
the QuantX Dataloader or Massive REST APIs used elsewhere in this project; see
`results/RESULTS.md` for what this implies for that experiment.

## Layout

```
src/            flat library modules (kernels, Markovian lift, simulation,
                moment bounds, calibration, configuration)
exp/            one script per experiment, run as `python -m exp.exp_N`
tests/          pytest unit tests of the src/ components
results/        aggregate metrics (.json/.csv) and static .png figures
data/           data-preparation code (empty package: no external data used)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are required for any experiment implemented here.

## Running the tests

```bash
python -m pytest -m "not market_data"
```

(there are no `market_data`-marked tests in this repository, since no
experiment loads market data; the flag is included for consistency.)

## Running the experiments

```bash
python -m exp.exp_1   # Table 2 (critical theta_c) and Figure 2 (second moment)
python -m exp.exp_2   # Markovian-lift simulation engine: discrete martingale checks
python -m exp.exp_3   # Table 3 / Figure 3(a): martingale property under positive correlation
python -m exp.exp_4   # Figure 3(b): time-averaged second-moment bounds under quadratic drift
python -m exp.exp_5   # Table 4 self-test of the calibration machinery (see Not reproduced)
```

Each script writes its metrics (`.json`) and static figures (`.png`) directly
into `results/`, overwriting the previous run's files. `results/RESULTS.md`
must be checked against the newly written files and updated by hand after a
rerun.

`reproduce.sh` runs the test suite followed by all five experiments in order.

## Experiment summary

1. **Linear-drift second-moment stability** (`exp/exp_1.py`): resolvent of the
   Riemann-Liouville fractional kernel, stability parameter gamma, critical
   theta_c and one-day innovation proxy nu (Table 2), conditional mean/second
   moment and their long-horizon behaviour (Figure 2), and the exponential
   growth rate above threshold.
2. **Markovian-lift simulation engine** (`exp/exp_2.py`): N-factor
   sum-of-exponentials approximation of the fractional kernel, symmetric
   operator-splitting scheme, and the discrete martingale property check that
   underlies experiments 3-5.
3. **Martingale property test** (`exp/exp_3.py`): sample mean of the raw
   conditional forward across kappa_2 values and grid refinements (Table 3,
   Figure 3(a)).
4. **Time-averaged second-moment bounds** (`exp/exp_4.py`): exact linear-drift
   moments, the Proposition 3.7 bound, and Monte Carlo estimates under
   quadratic drift (Figure 3(b)).
5. **Smile calibration** (`exp/exp_5.py`): calibration machinery self-test
   against the paper's own published Table 4 parameter vectors; the paper's
   Bitcoin/Ether option data could not be obtained (see `results/RESULTS.md`).

See `results/RESULTS.md` for the full numerical comparison against the
paper's reported values, verdicts, deviations and what was not reproduced.
