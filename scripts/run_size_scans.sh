#!/usr/bin/env bash
# Simulations of Extension II with the C implementation. Run from the repository root.
# The full set takes about one hour with 24 worker processes.
#   1  size scan: 19 space sizes, 100 seeds, 20,000 steps
#   2  pattern counts at L=200 with all metrics recorded at every step, 40 seeds
#   3  configurations for the snapshot panels (L=200 seed 0; L=400 seed 5)
#   4  sensitivity to the mutation and death probabilities: 10,000 steps, 30 seeds
#   5  longer horizon: L=240, 320 and 400, 40,000 steps, runs stopped at a mean component size of 2,000
#   6  200 runs at L=100 for the comparison with the JAX implementation
set -eu
cd "$(dirname "$0")/.."
PY="uv run --no-sync python"
W=24
run() {
    echo "== $(date '+%F %T') $*"
    "$@"
}

# 1
run $PY scripts/run_scan_cpu.py --sizes 100 120 140 160 180 200 210 220 230 240 250 260 280 300 320 340 360 380 400 \
    --seeds 0-99 --workers $W --output-root results/cpu_scan_full

# 2
run $PY scripts/run_scan_cpu.py --sizes 200 --seeds 0-39 --sample-interval 1 --workers $W \
    --output-root results/cpu_novelty

# 3
for T in 5000 10000; do
    run $PY scripts/run_scan_cpu.py --sizes 200 --seeds 0 --steps $T --workers 1 \
        --output-root results/cpu_snapshots/t$T
done
for T in 2000 5000 10000; do
    run $PY scripts/run_scan_cpu.py --sizes 400 --seeds 5 --steps $T --workers 1 \
        --output-root results/cpu_snapshots/t$T
done

# 4
SENS="--sizes 100 200 240 300 320 400 --seeds 0-29 --steps 10000 --workers $W --output-root results/cpu_sensitivity"
for MU in 0.001 0.002 0.005 0.01; do run $PY scripts/run_scan_cpu.py $SENS --mu $MU; done
for PD in 0.0005 0.001 0.005 0.01; do run $PY scripts/run_scan_cpu.py $SENS --death-prob $PD; done

# 5
run $PY scripts/run_scan_cpu.py --sizes 240 320 400 --seeds 0-99 --steps 40000 --stop-mean-size 2000 --workers $W \
    --output-root results/cpu_scan_40k

# 6
run $PY scripts/run_scan_cpu.py --sizes 100 --seeds 0-199 --workers $W --output-root results/cpu_scan_check

echo "== $(date '+%F %T') all jobs done"
