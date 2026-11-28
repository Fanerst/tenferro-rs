#!/bin/bash
# Step-172 measurement on itp: CPU, 16 threads, complex64 (tensor4all/tenferro-rs#2756).
# Unpinned, as the earlier itp CPU runs (review-2739, confirm-2744).
# Usage: run.sh ROUNDS WARM
set -u
rounds=$1; warm=$2
export RAYON_NUM_THREADS=16 OMP_NUM_THREADS=16
cd ~/claude_runs/step172
bin=./step172.bin; case=case_n30m14.txt
mkdir -p results
run() { # mode
  local t0=$(date +%s.%N)
  /usr/bin/time -v "$bin" "$case" "$1" 16 "$warm" >> results/runs.jsonl 2>> results/time.log
  local rc=$?
  echo "$(date -u +%FT%TZ) $1 exit $rc $(echo "$(date +%s.%N) - $t0" | bc) s load $(cut -d' ' -f1 /proc/loadavg)" >> results/progress.log
}
orders=("v0 pair row" "pair row v0" "row v0 pair")
for r in $(seq 1 "$rounds"); do
  for m in ${orders[$(( (r - 1) % 3 ))]}; do run "$m"; done
done
echo "done $(date -u +%FT%TZ)" >> results/progress.log
