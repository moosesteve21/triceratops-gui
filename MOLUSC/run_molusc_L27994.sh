#!/bin/bash
# MOLUSC for L 279-94 (TIC 100796276): bound companions still allowed by
# Gaia (RUWE + Gaia imaging contrast).  Run in WSL with the environment that
# runs the pipeline (triplus):   bash run_molusc_L27994.sh
# Needs internet: the Gaia archive (gea.esac.esa.int) or, as fallback,
# VizieR (Gaia DR3, I/355/gaiadr3).
# 1,000,000 companions take about 10 minutes.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../TIC100796276_molusc"
mkdir -p "$OUT"
cd "$HERE/code"            # MOLUSC reads its model tables from this folder
N=1000000
python BinaryStarGUI.py -v --age 5 --ruwe --gaia \
    "$OUT/L27994_N$N" 20h47m04.89s -- -53d08m01.8s $N 0.403 | tee "$OUT/molusc_log.txt"
echo
echo "Done: $OUT/L27994_N${N}_kept.csv"
echo "Pipeline:  --molusc-file $OUT/L27994_N${N}_kept.csv --molusc-n $N"
