geometry: {'pairs': 423, 'rows_left': 32, 'rows_right': 16, 'distinct_left_rows_used': 32, 'm': 1048576, 'k': 32, 'n': 2, 'chunk': 2, 'chunks': 212, 'roles': 'lhs MMMMMMMMMMKMKMKMMMMMMKPKMM rhs KKPKKKN out MMMMMMMMMMMMMMMMMMMMNP'}

| variant | cold s (median, range) | warm s (median, range) | warm, reused output s | n cold / warm |
|---|---|---|---|---|
| pair | 6.756 (6.756–6.756) | 5.559 (5.255–5.794) | 3.783 (3.650–3.784) | 1 / 3 |
| row | 4.054 (4.054–4.054) | 3.069 (3.006–3.096) | 0.638 (0.630–0.638) | 1 / 3 |

pair: gathered_rows 0, gathered_elements 0, gemm_calls 423, shape (1048576, 32, cols) with cols:count 2:423
    jobs (cold) median 0.000021 s, GroupedPlan::new (cold) median 0.000160 s, child plans 1, route Inner { threads: 16 }
    warm exec median 4.599 s, gather median 0.000000 s
    peak RSS (VmHWM) max 14.62 GiB
row: gathered_rows 423, gathered_elements 27072, gemm_calls 32, shape (1048576, 32, cols) with cols:count 20:1 22:3 24:7 26:8 28:4 30:8 32:1
    jobs (cold) median 0.000032 s, GroupedPlan::new (cold) median 0.000183 s, child plans 7, route Inner { threads: 16 }
    warm exec median 1.720 s, gather median 0.000064 s
    peak RSS (VmHWM) max 14.66 GiB
