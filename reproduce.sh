#!/usr/bin/env bash
# Reproduce the replication of Sepp & Rakhmonov (2026), "Rough log-normal
# stochastic volatility with quadratic drift: second-moment stability and the
# martingale property under positive correlation".
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== tests ==="
python -m pytest -m "not market_data" || exit 1

echo "=== exp_1: linear-drift second-moment stability ==="
python -m exp.exp_1 || exit 1

echo "=== exp_2: Markovian-lift simulation engine ==="
python -m exp.exp_2 || exit 1

echo "=== exp_3: martingale property test ==="
python -m exp.exp_3 || exit 1

echo "=== exp_4: time-averaged second-moment bounds ==="
python -m exp.exp_4 || exit 1

echo "=== exp_5: smile calibration self-test ==="
python -m exp.exp_5 || exit 1

echo "=== done: see results/RESULTS.md ==="
