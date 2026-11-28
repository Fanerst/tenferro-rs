"""Summarise the step-172 runs (tensor4all/tenferro-rs#2756).

Reads the `STEP172 {json}` lines of results/runs.jsonl (one process per variant per
round) and prints, per variant: cold and warm medians with ranges, the planning and
gather parts, the gathered-row counters, the GEMM calls and shapes, the route and the
peak RSS. Usage: python3 analyze.py results/runs.jsonl
"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    rows = []
    for line in open(path):
        if line.startswith("STEP172 "):
            rows.append(json.loads(line[len("STEP172 "):]))
    return rows


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rng(xs):
    return f"{min(xs):.3f}–{max(xs):.3f}" if xs else "–"


def main():
    rows = load(sys.argv[1])
    info = next((r for r in rows if r["variant"] == "info"), None)
    if info:
        print("geometry:", {k: info[k] for k in ("pairs", "rows_left", "rows_right",
              "distinct_left_rows_used", "m", "k", "n", "chunk", "chunks", "roles")})
    by = defaultdict(lambda: defaultdict(list))
    extra = {}
    for r in rows:
        v = r["variant"]
        if v in ("info", "relayout", "engine") or "kind" not in r:
            continue
        by[v][r["kind"]].append(r)
        extra.setdefault(v, r)
    print()
    print("| variant | cold s (median, range) | warm s (median, range) | warm, reused output s | n cold / warm |")
    print("|---|---|---|---|---|")
    for v in ("v0", "pair", "row"):
        if v not in by:
            continue
        cold = [r["seconds"] for r in by[v]["cold"]]
        warm = [r["seconds"] for r in by[v]["warm"]]
        reuse = [r["seconds"] for r in by[v].get("warm-reused-output", [])]
        print(f"| {v} | {med(cold):.3f} ({rng(cold)}) | {med(warm):.3f} ({rng(warm)}) | "
              f"{(f'{med(reuse):.3f} ({rng(reuse)})') if reuse else '–'} | {len(cold)} / {len(warm)} |")
    print()
    for v in ("v0", "pair", "row"):
        if v not in by:
            continue
        r = extra[v]
        cold = by[v]["cold"]
        print(f"{v}: gathered_rows {r['gathered_rows']}, gathered_elements {r['gathered_elements']}, "
              f"gemm_calls {r['gemm_calls']}, shape {r['gemm_shape']}")
        if v == "v0":
            print(f"    prep (ConcreteEinsumPlan, cold) median {med([c['prep_seconds'] for c in cold]):.6f} s")
        else:
            print(f"    jobs (cold) median {med([c['jobs_seconds'] for c in cold]):.6f} s, "
                  f"GroupedPlan::new (cold) median {med([c['plan_seconds'] for c in cold]):.6f} s, "
                  f"child plans {r['child_plans']}, route {r['route']}")
            warm = by[v]["warm"]
            print(f"    warm exec median {med([w['exec_seconds'] for w in warm]):.3f} s, "
                  f"gather median {med([w['gather_seconds'] for w in warm]):.6f} s")
        hwm = [x["vm_hwm_kib"] for x in by[v]["cold"] + by[v]["warm"]]
        print(f"    peak RSS (VmHWM) max {max(hwm) / 2**20:.2f} GiB")
    for r in rows:
        if r["variant"] == "relayout":
            print(f"relayout: left transpose {r['left_transpose_seconds']} s for {r['left_elements']} elements")
        if "rel_diff_vs_v0" in r:
            print(f"check {r['variant']}: rel diff vs v0 {r['rel_diff_vs_v0']}")


if __name__ == "__main__":
    main()
