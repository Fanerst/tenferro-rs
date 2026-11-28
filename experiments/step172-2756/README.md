# Step 172 of n30m14, three pair-step lowerings (tensor4all/tenferro-rs#2756)

Measurement only, not for merge. The harness is the `step172-exp` feature of
`tenferro-einsum` (`crates/tenferro-einsum/src/sampled/exec/step172.rs`), on `main`
`61e0a108`:

- `v0`: `execute_concrete` on a one-step program holding step 172's own `PairStep`;
- `pair`: one `cpueinsum::GroupedPlan` job per pair, both children in place;
- `row`: pairs ordered by left row, one job per distinct left row, only the small side gathered.

Files:

- `make_case.py DATA_DIR case.txt` writes n30m14 (`data/n30m14` of Fanerst/sampled-einsum)
  in the harness's plain-text format.
- `bin/` is the driver; it expects this tree next to it as `../tf`.
- `run.sh ROUNDS WARM` runs each variant in its own process, rotating the order.
- `analyze.py results/runs.jsonl` prints the medians.
- `results/`: the timing rounds (`runs.jsonl`, `time.log`, `progress.log`; built from
  `504dd5e9`), the correctness check (`check.*`; its grouped timings include the comparison
  and are not timings), and the diagnostics (`diag.*`; forced `Inner` strategy and one job
  per column count on one thread; built from `888dc28c`).

Host: itp (2× Xeon Platinum 8268, 96 threads, shared; load 35–40 during the runs), 16
threads, complex64, unpinned.
