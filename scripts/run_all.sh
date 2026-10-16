#!/usr/bin/env bash
# Full reproduction pipeline. Expect many CPU/GPU hours for the complete grid
# (3 systems x 3 penetrations x 7 methods x 10 seeds x 30 days x 96 intervals);
# use --n-jobs and the --days / --seeds options to split the work.
set -euo pipefail
cd "$(dirname "$0")/.."
J=${N_JOBS:-8}

python scripts/prepare_data.py --rebuild
python scripts/train_pinn.py --variant qpong data_only pinn_opf --n-jobs "$J"
python scripts/run_quantum_standalone.py --table V VI
python scripts/run_comparison.py --n-jobs "$J"
python scripts/run_comparison.py --system ieee118 --stress normal high_demand low_renewables combined --tag stress --n-jobs "$J"
python scripts/run_ablation.py --n-jobs "$J"
python scripts/run_settlement.py --n-jobs "$J"
python scripts/run_noise.py --n-jobs "$J"
# hardware (needs IBM Quantum access):
#   python scripts/run_hardware.py train  && python scripts/run_hardware.py submit && python scripts/run_hardware.py replay
python scripts/make_tables.py
python scripts/make_figures.py
