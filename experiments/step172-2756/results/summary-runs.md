geometry: {'pairs': 423, 'rows_left': 32, 'rows_right': 16, 'distinct_left_rows_used': 32, 'm': 1048576, 'k': 32, 'n': 2, 'chunk': 2, 'chunks': 212, 'roles': 'lhs MMMMMMMMMMKMKMKMMMMMMKPKMM rhs KKPKKKN out MMMMMMMMMMMMMMMMMMMMNP'}

| variant | cold s (median, range) | warm s (median, range) | warm, reused output s | n cold / warm |
|---|---|---|---|---|
| v0 | 65.254 (61.638–69.367) | 66.856 (63.249–70.315) | – | 3 / 9 |
| pair | 4.406 (3.828–5.042) | 4.401 (4.169–5.003) | 2.211 (1.960–4.011) | 3 / 9 |
| row | 2.236 (1.807–2.408) | 2.425 (2.150–2.660) | 0.700 (0.637–1.049) | 3 / 9 |

v0: gathered_rows 846, gathered_elements 14193551808, gemm_calls 212, shape 212 batched calls of (1048576, 32, 2), batch 2 (last 1)
    prep (ConcreteEinsumPlan, cold) median 0.000158 s
    peak RSS (VmHWM) max 15.68 GiB
pair: gathered_rows 0, gathered_elements 0, gemm_calls 423, shape (1048576, 32, cols) with cols:count 2:423
    jobs (cold) median 0.000023 s, GroupedPlan::new (cold) median 0.000156 s, child plans 1, route Outer { lanes: 16 }
    warm exec median 3.481 s, gather median 0.000000 s
    peak RSS (VmHWM) max 14.62 GiB
row: gathered_rows 423, gathered_elements 27072, gemm_calls 32, shape (1048576, 32, cols) with cols:count 20:1 22:3 24:7 26:8 28:4 30:8 32:1
    jobs (cold) median 0.000069 s, GroupedPlan::new (cold) median 0.000187 s, child plans 7, route Outer { lanes: 16 }
    warm exec median 1.344 s, gather median 0.000065 s
    peak RSS (VmHWM) max 14.65 GiB
