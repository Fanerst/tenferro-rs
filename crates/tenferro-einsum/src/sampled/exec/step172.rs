//! Measurement for tensor4all/tenferro-rs#2756, not for merge.
//!
//! Step 172 of the n30m14 circuit is the gathered-pairs step that holds 87% of
//! v0's gathered elements. This module runs it three ways on one CPU engine:
//!
//! - `v0`: the shipping executor, [`execute_concrete`] on a one-step program,
//!   which gathers both children's rows for every chunk;
//! - `pair`: one [`GroupedPlan`] job per pair, over both children in place;
//! - `row`: the pairs ordered by left row, one job per distinct left row over
//!   all of that row's pairs, with only the small side gathered.
//!
//! The two grouped variants read the children in the layout a v1 planner would
//! have their producers write (`[M, K, P]` and `[K, N, P]`, first axis
//! fastest). `check` runs all three on the same operands, transposing v0's
//! children into that layout once, and compares the results.

use std::collections::{BTreeMap, HashMap};
use std::mem::MaybeUninit;
use std::time::Instant;

use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{GroupedGemmJob, GroupedPlan, GroupedRoute};
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::{Complex32, DType, Tensor, TensorRead, TensorStructural};

use super::super::plan::{Contraction, Instr, PairStep, Program};
use super::super::{SampleOptions, SampledEinsumPlan, Samples};
use super::execute_concrete;
use crate::api::ConjMask;
use crate::concrete::ConcreteEinsumInputSpec;
use crate::{ConcreteEinsumPlan, ContractionTree, EinsumOptimize, EinsumOptions, Subscripts};

type Res<T> = std::result::Result<T, String>;

fn err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

struct Tokens<'a>(std::str::SplitWhitespace<'a>);

impl Tokens<'_> {
    fn word(&mut self) -> Res<&str> {
        self.0
            .next()
            .ok_or_else(|| "unexpected end of the case file".to_string())
    }

    fn tag(&mut self, want: &str) -> Res<()> {
        let w = self.word()?;
        if w == want {
            Ok(())
        } else {
            Err(format!("expected `{want}`, found `{w}`"))
        }
    }

    fn num<T: std::str::FromStr>(&mut self) -> Res<T>
    where
        T::Err: std::fmt::Display,
    {
        let w = self.word()?.to_string();
        w.parse().map_err(|e: T::Err| format!("`{w}`: {e}"))
    }

    fn nums<T: std::str::FromStr>(&mut self, len: usize) -> Res<Vec<T>>
    where
        T::Err: std::fmt::Display,
    {
        (0..len).map(|_| self.num()).collect()
    }
}

/// The network, path and samples written by `make_case.py`; every extent is 2.
struct Case {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    path: Vec<(usize, usize)>,
    labels: Vec<u32>,
    n: usize,
    values: Vec<usize>,
}

fn read_case(path: &str) -> Res<Case> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut t = Tokens(text.split_whitespace());
    t.tag("inputs")?;
    let count: usize = t.num()?;
    let mut inputs = Vec::with_capacity(count);
    for _ in 0..count {
        let len: usize = t.num()?;
        inputs.push(t.nums(len)?);
    }
    t.tag("output")?;
    let len: usize = t.num()?;
    let output = t.nums(len)?;
    t.tag("path")?;
    let steps: usize = t.num()?;
    let mut path = Vec::with_capacity(steps);
    for _ in 0..steps {
        path.push((t.num()?, t.num()?));
    }
    t.tag("samples")?;
    let n: usize = t.num()?;
    let width: usize = t.num()?;
    let labels = t.nums(width)?;
    let values = t.nums(n * width)?;
    Ok(Case {
        inputs,
        output,
        path,
        labels,
        n,
        values,
    })
}

fn build_plan(case: &Case) -> Res<SampledEinsumPlan> {
    let sizes: HashMap<u32, usize> = case.inputs.iter().flatten().map(|&l| (l, 2)).collect();
    let input_refs: Vec<&[u32]> = case.inputs.iter().map(Vec::as_slice).collect();
    let subscripts = Subscripts::new(&input_refs, &case.output);
    let shapes: Vec<Vec<usize>> = case
        .inputs
        .iter()
        .map(|ls| ls.iter().map(|l| sizes[l]).collect())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let tree = ContractionTree::from_pairs(&subscripts, &shape_refs, &case.path).map_err(err)?;
    let options = SampleOptions::default()
        .einsum(EinsumOptions::default().optimize(EinsumOptimize::Tree(tree)))
        .max_chunk_elements(1 << 26)
        .outer_select_ratio(2.0);
    let samples = Samples::new(case.labels.iter().copied(), case.n, case.values.clone());
    let meta = shapes.iter().map(|s| (DType::C32, s.clone())).collect();
    SampledEinsumPlan::plan(meta, &subscripts, &samples, &options).map_err(err)
}

/// One gathered-pairs step, and its children's roles.
struct Geometry {
    step: PairStep,
    full: Contraction,
    tail: Option<Contraction>,
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
    rows_left: usize,
    rows_right: usize,
    /// Axis `i` of the `[M, K, P]` left operand is axis `perm_left[i]` of v0's.
    perm_left: Vec<usize>,
    perm_right: Vec<usize>,
    roles: String,
}

fn geometry(plan: &SampledEinsumPlan, which: usize) -> Res<Geometry> {
    let program = &plan.program;
    let step = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            Instr::Pairs(step) => Some((**step).clone()),
            _ => None,
        })
        .nth(which)
        .ok_or_else(|| format!("the plan has no pairs step {which}"))?;
    let full = program.contractions[step.full].clone();
    let tail = step.tail.map(|t| program.contractions[t].clone());
    let lhs = full.subscripts.inputs[0].clone();
    let rhs = full.subscripts.inputs[1].clone();
    let out = full.subscripts.output.clone();
    let p = lhs[step.axis_left];
    if rhs[step.axis_right] != p || out[step.out_axis] != p {
        return Err("the pair label is not on both children and the output".into());
    }
    let left_shape = program.slot_shapes[step.left].clone();
    let right_shape = program.slot_shapes[step.right].clone();
    let mut extent: HashMap<u32, usize> = HashMap::new();
    for (labels, shape) in [(&lhs, &left_shape), (&rhs, &right_shape)] {
        for (l, d) in labels.iter().zip(shape.iter()) {
            if *l != p {
                extent.insert(*l, *d);
            }
        }
    }
    let m_labels: Vec<u32> = lhs
        .iter()
        .copied()
        .filter(|x| *x != p && !rhs.contains(x))
        .collect();
    let k_labels: Vec<u32> = lhs
        .iter()
        .copied()
        .filter(|x| *x != p && rhs.contains(x) && !out.contains(x))
        .collect();
    let kept_shared = lhs
        .iter()
        .filter(|x| **x != p && rhs.contains(x) && out.contains(x))
        .count();
    let n_labels: Vec<u32> = rhs
        .iter()
        .copied()
        .filter(|x| *x != p && !lhs.contains(x))
        .collect();
    if kept_shared != 0 {
        return Err("kept shared labels (|B| > 1) are not handled here".into());
    }
    if m_labels.iter().chain(&n_labels).any(|x| !out.contains(x)) {
        return Err("a label is summed on one side only".into());
    }
    let expect: Vec<u32> = m_labels
        .iter()
        .chain(&n_labels)
        .copied()
        .chain([p])
        .collect();
    if out != expect {
        return Err(format!("v0's output order {out:?} is not [M, N, P] {expect:?}"));
    }
    let product = |ls: &[u32]| ls.iter().map(|l| extent[l]).product::<usize>();
    let pos = |labels: &[u32], x: u32| labels.iter().position(|y| *y == x).expect("label");
    let perm_left: Vec<usize> = m_labels
        .iter()
        .chain(&k_labels)
        .copied()
        .chain([p])
        .map(|x| pos(&lhs, x))
        .collect();
    let perm_right: Vec<usize> = k_labels
        .iter()
        .chain(&n_labels)
        .copied()
        .chain([p])
        .map(|x| pos(&rhs, x))
        .collect();
    let role = |labels: &[u32]| -> String {
        labels
            .iter()
            .map(|x| {
                if *x == p {
                    'P'
                } else if m_labels.contains(x) {
                    'M'
                } else if n_labels.contains(x) {
                    'N'
                } else {
                    'K'
                }
            })
            .collect()
    };
    let roles = format!("lhs {} rhs {} out {}", role(&lhs), role(&rhs), role(&out));
    Ok(Geometry {
        m: product(&m_labels),
        k: product(&k_labels),
        n: product(&n_labels),
        rows_left: left_shape[step.axis_left],
        rows_right: right_shape[step.axis_right],
        step,
        full,
        tail,
        left_shape,
        right_shape,
        perm_left,
        perm_right,
        roles,
    })
}

fn random_c32(len: usize, seed: u64) -> Vec<Complex32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let a = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let b = ((s >> 16) & 0xFF_FFFF) as f32 / (1u64 << 24) as f32 - 0.5;
        v.push(Complex32::new(a, b));
    }
    v
}

fn status_kib(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// Largest absolute difference over the largest absolute reference value.
fn rel_diff(got: &[Complex32], want: &[Complex32]) -> f64 {
    let mut diff = 0f64;
    let mut scale = 0f64;
    for (g, w) in got.iter().zip(want) {
        diff = diff.max(f64::from((g - w).norm()));
        scale = scale.max(f64::from(w.norm()));
    }
    diff / scale.max(f64::MIN_POSITIVE)
}

fn emit(fields: &[(&str, String)]) {
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    println!("STEP172 {{{}}}", body.join(","));
}

fn q(s: &str) -> String {
    format!("\"{s}\"")
}

/// v0 on a one-step program: `warm + 1` runs, the first with fresh plans.
fn run_v0(
    engine: &mut CpuEngine,
    g: &Geometry,
    left: &Tensor,
    right: &Tensor,
    warm: usize,
    tag: &str,
) -> Res<Tensor> {
    let mut step = g.step.clone();
    step.left = 0;
    step.right = 1;
    step.out = 2;
    step.full = 0;
    step.tail = g.tail.as_ref().map(|_| 1);
    let chunks = step.chunks();
    let contractions: Vec<Contraction> =
        std::iter::once(g.full.clone()).chain(g.tail.clone()).collect();
    let t = Instant::now();
    let plans = contractions
        .iter()
        .map(|c| {
            let specs = c
                .input_shapes
                .iter()
                .map(|s| ConcreteEinsumInputSpec::new(DType::C32, s.clone()))
                .collect();
            ConcreteEinsumPlan::prepare_planned(
                specs,
                &c.subscripts,
                &EinsumOptimize::False,
                ConjMask::default(),
            )
        })
        .collect::<crate::Result<Vec<_>>>()
        .map_err(err)?;
    let prep = t.elapsed().as_secs_f64();
    let program = Program {
        n_slots: 3,
        instrs: vec![Instr::Pairs(Box::new(step))],
        contractions,
        result: 2,
        output_shape: g.step.out_shape.clone(),
        leaf_rows: Vec::new(),
        steps: Vec::new(),
        slot_shapes: vec![
            g.left_shape.clone(),
            g.right_shape.clone(),
            g.step.out_shape.clone(),
        ],
    };
    let reads = [TensorRead::from_tensor(left), TensorRead::from_tensor(right)];
    let pairs = g.step.rows();
    let gathered_elems = pairs * (g.m * g.k + g.k * g.n);
    let mut last = None;
    for run in 0..=warm {
        // Drop the previous run's result before this run allocates its own.
        drop(last.take());
        let t = Instant::now();
        let (result, stats) = execute_concrete(&program, &plans, &reads, engine).map_err(err)?;
        let secs = t.elapsed().as_secs_f64();
        emit(&[
            ("tag", q(tag)),
            ("variant", q("v0")),
            ("run", run.to_string()),
            ("kind", q(if run == 0 { "cold" } else { "warm" })),
            ("seconds", format!("{secs:.6}")),
            ("prep_seconds", format!("{:.6}", if run == 0 { prep } else { 0.0 })),
            ("gathered_rows", (2 * pairs).to_string()),
            ("gathered_elements", gathered_elems.to_string()),
            ("gemm_calls", chunks.to_string()),
            (
                "gemm_shape",
                q(&format!(
                    "{chunks} batched calls of ({}, {}, {}), batch {} (last {})",
                    g.m,
                    g.k,
                    g.n,
                    g.step.chunk,
                    pairs - (chunks - 1) * g.step.chunk
                )),
            ),
            ("chunks", stats.chunks.to_string()),
            ("peak_elements", stats.peak_elements.to_string()),
            ("vm_hwm_kib", status_kib("VmHWM").to_string()),
        ]);
        last = Some(result);
    }
    last.ok_or_else(|| "no run".to_string())
}

/// The grouped variants. `by_row` orders the pairs by left row and runs one job
/// per distinct left row; otherwise one job per pair. Each of `warm + 1` runs
/// takes a fresh output; `reuse` more runs then write one output again.
#[allow(clippy::too_many_arguments)]
fn run_grouped(
    engine: &mut CpuEngine,
    g: &Geometry,
    left: &[Complex32],
    right: &[Complex32],
    by_row: bool,
    warm: usize,
    reuse: usize,
    reference: Option<&[Complex32]>,
    tag: &str,
) -> Res<()> {
    let (m, k, n) = (g.m, g.k, g.n);
    let pairs = g.step.rows();
    let one = Complex32::new(1.0, 0.0);
    let zero = Complex32::new(0.0, 0.0);
    let variant = if by_row { "row" } else { "pair" };

    let t = Instant::now();
    // Pair order of the output, and the jobs.
    let order: Vec<usize> = if by_row {
        let mut o: Vec<usize> = (0..pairs).collect();
        o.sort_by_key(|&p| (g.step.left_rows[p], p));
        o
    } else {
        (0..pairs).collect()
    };
    let jobs: Vec<GroupedGemmJob> = if by_row {
        let mut jobs = Vec::new();
        let mut start = 0;
        while start < pairs {
            let row = g.step.left_rows[order[start]];
            let mut end = start + 1;
            while end < pairs && g.step.left_rows[order[end]] == row {
                end += 1;
            }
            let t_len = end - start;
            jobs.push(GroupedGemmJob::new(
                row * m * k,
                start * k * n,
                start * m * n,
                m,
                k,
                n * t_len,
            ));
            start = end;
        }
        jobs
    } else {
        (0..pairs)
            .map(|p| {
                GroupedGemmJob::new(
                    g.step.left_rows[p] * m * k,
                    g.step.right_rows[p] * k * n,
                    p * m * n,
                    m,
                    k,
                    n,
                )
            })
            .collect()
    };
    let jobs_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let plan = GroupedPlan::<Complex32>::new(&jobs, &PlanConfig::default()).map_err(err)?;
    let plan_seconds = t.elapsed().as_secs_f64();
    let mut cols: BTreeMap<usize, usize> = BTreeMap::new();
    for job in &jobs {
        *cols.entry(job.cols).or_default() += 1;
    }
    let cols_hist: Vec<String> = cols.iter().map(|(c, x)| format!("{c}:{x}")).collect();

    let out_len = pairs * m * n;
    let mut gathered: Vec<Complex32> = if by_row {
        vec![zero; pairs * k * n]
    } else {
        Vec::new()
    };
    let mut kept: Option<Box<[MaybeUninit<Complex32>]>> = None;
    for run in 0..=(warm + reuse) {
        let reusing = run > warm;
        let t_all = Instant::now();
        // The small side's rows in output order (the row variant only).
        let t = Instant::now();
        if by_row {
            for (j, &p) in order.iter().enumerate() {
                let r = g.step.right_rows[p];
                gathered[j * k * n..(j + 1) * k * n]
                    .copy_from_slice(&right[r * k * n..(r + 1) * k * n]);
            }
        }
        let gather_seconds = t.elapsed().as_secs_f64();
        // A fresh run drops the previous output before allocating its own.
        let mut out = if reusing {
            kept.take().ok_or_else(|| "no output to reuse".to_string())?
        } else {
            drop(kept.take());
            Box::<[Complex32]>::new_uninit_slice(out_len)
        };
        let rhs: &[Complex32] = if by_row { &gathered } else { right };
        let t = Instant::now();
        let (route, diff) = engine
            .with_exec(|exec| {
                let (written, route) = plan.execute_uninit(exec, one, left, rhs, zero, &mut out)?;
                let diff = reference.map(|want| {
                    if by_row {
                        let mut worst = 0f64;
                        for (j, &p) in order.iter().enumerate() {
                            worst = worst.max(rel_diff(
                                &written[j * m * n..(j + 1) * m * n],
                                &want[p * m * n..(p + 1) * m * n],
                            ));
                        }
                        worst
                    } else {
                        rel_diff(written, want)
                    }
                });
                Ok::<(GroupedRoute, Option<f64>), cpueinsum::Error>((route, diff))
            })
            .map_err(err)?
            .map_err(err)?;
        let exec_seconds = t.elapsed().as_secs_f64();
        let secs = t_all.elapsed().as_secs_f64();
        let cold = run == 0;
        let total = if cold {
            secs + jobs_seconds + plan_seconds
        } else {
            secs
        };
        let kind = if cold {
            "cold"
        } else if reusing {
            "warm-reused-output"
        } else {
            "warm"
        };
        let mut fields = vec![
            ("tag", q(tag)),
            ("variant", q(variant)),
            ("run", run.to_string()),
            ("kind", q(kind)),
            ("seconds", format!("{total:.6}")),
            ("exec_seconds", format!("{exec_seconds:.6}")),
            ("gather_seconds", format!("{gather_seconds:.6}")),
            ("jobs_seconds", format!("{:.6}", if cold { jobs_seconds } else { 0.0 })),
            ("plan_seconds", format!("{:.6}", if cold { plan_seconds } else { 0.0 })),
            ("gathered_rows", (if by_row { pairs } else { 0 }).to_string()),
            ("gathered_elements", (if by_row { pairs * k * n } else { 0 }).to_string()),
            ("gemm_calls", jobs.len().to_string()),
            (
                "gemm_shape",
                q(&format!("({m}, {k}, cols) with cols:count {}", cols_hist.join(" "))),
            ),
            ("child_plans", cols.len().to_string()),
            ("route", q(&format!("{route:?}"))),
            ("vm_hwm_kib", status_kib("VmHWM").to_string()),
        ];
        if let Some(d) = diff {
            fields.push(("rel_diff_vs_v0", format!("{d:.3e}")));
        }
        emit(&fields);
        kept = Some(out);
    }
    Ok(())
}

/// The engine; `STEP172_POLICY` (`inner`, `outer` or `sequential`) forces its
/// batch strategy, which decides how a grouped plan splits its jobs. Unset
/// keeps the shipped default, `Auto`.
fn engine(threads: usize) -> Res<(CpuEngine, String)> {
    use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
    let pool = Pool::builder().threads(threads).build().map_err(err)?;
    let mut engine = CpuEngine::new(pool);
    let policy = std::env::var("STEP172_POLICY").unwrap_or_default();
    let strategy = match policy.as_str() {
        "" | "auto" => None,
        "inner" => Some(CpuBatchStrategy::Inner),
        "outer" => Some(CpuBatchStrategy::Outer),
        "sequential" => Some(CpuBatchStrategy::Sequential),
        other => return Err(format!("unknown STEP172_POLICY `{other}`")),
    };
    if let Some(strategy) = strategy {
        engine.set_batch_policy(CpuBatchPolicy::new(strategy));
    }
    let name = format!("{:?}", engine.batch_policy().strategy);
    Ok((engine, name))
}

/// Entry point of the step-172 measurement (feature `step172-exp`).
///
/// `args`: `CASE MODE THREADS WARM [WHICH]`, where `MODE` is `info`, `v0`,
/// `pair`, `row`, `check` or `kernel`, and `WHICH` picks the gathered-pairs step (0 is
/// step 172 of n30m14). Each result is one `STEP172 {json}` line on stdout.
///
/// # Errors
///
/// A message for a malformed case, a step this measurement does not cover, or
/// a failed run.
pub fn run(args: &[String]) -> std::result::Result<(), String> {
    let [case, mode, threads, warm, rest @ ..] = args else {
        return Err("usage: CASE info|v0|pair|row|check|kernel THREADS WARM [WHICH]".into());
    };
    let threads: usize = threads.parse().map_err(err)?;
    let warm: usize = warm.parse().map_err(err)?;
    let which: usize = rest.first().map_or(Ok(0), |w| w.parse()).map_err(err)?;
    let tag = format!("{mode}-t{threads}");
    let t = Instant::now();
    let case = read_case(case)?;
    let plan = build_plan(&case)?;
    let plan_seconds = t.elapsed().as_secs_f64();
    let g = geometry(&plan, which)?;
    let pairs = g.step.rows();
    let distinct_left: std::collections::BTreeSet<usize> =
        g.step.left_rows.iter().copied().collect();
    emit(&[
        ("tag", q(&tag)),
        ("variant", q("info")),
        ("sampled_plan_seconds", format!("{plan_seconds:.6}")),
        ("pairs", pairs.to_string()),
        ("rows_left", g.rows_left.to_string()),
        ("rows_right", g.rows_right.to_string()),
        ("distinct_left_rows_used", distinct_left.len().to_string()),
        ("m", g.m.to_string()),
        ("k", g.k.to_string()),
        ("n", g.n.to_string()),
        ("chunk", g.step.chunk.to_string()),
        ("chunks", g.step.chunks().to_string()),
        ("left_shape_len", g.left_shape.len().to_string()),
        ("roles", q(&g.roles)),
        ("threads", threads.to_string()),
    ]);
    if mode == "info" {
        return Ok(());
    }
    let (mut engine, policy) = engine(threads)?;
    emit(&[
        ("tag", q(&tag)),
        ("variant", q("engine")),
        ("batch_strategy", q(&policy)),
        ("buffer_pool_limit_bytes", engine.buffer_pool_limit_bytes().to_string()),
    ]);
    let left_len = g.m * g.k * g.rows_left;
    let right_len = g.k * g.n * g.rows_right;
    match mode.as_str() {
        "v0" => {
            let left = Tensor::from_vec_col_major(g.left_shape.clone(), random_c32(left_len, 1))
                .map_err(err)?;
            let right =
                Tensor::from_vec_col_major(g.right_shape.clone(), random_c32(right_len, 2))
                    .map_err(err)?;
            run_v0(&mut engine, &g, &left, &right, warm, &tag)?;
        }
        "pair" | "row" => {
            let left = random_c32(left_len, 1);
            let right = random_c32(right_len, 2);
            run_grouped(
                &mut engine,
                &g,
                &left,
                &right,
                mode == "row",
                warm,
                warm,
                None,
                &tag,
            )?;
        }
        "check" => {
            let left = Tensor::from_vec_col_major(g.left_shape.clone(), random_c32(left_len, 1))
                .map_err(err)?;
            let right =
                Tensor::from_vec_col_major(g.right_shape.clone(), random_c32(right_len, 2))
                    .map_err(err)?;
            let reference_tensor = run_v0(&mut engine, &g, &left, &right, 0, &tag)?;
            let t = Instant::now();
            let left1 = engine
                .transpose_read(TensorRead::from_tensor(&left), &g.perm_left)
                .map_err(err)?;
            let relayout_left = t.elapsed().as_secs_f64();
            let right1 = engine
                .transpose_read(TensorRead::from_tensor(&right), &g.perm_right)
                .map_err(err)?;
            let want_left: Vec<usize> = g.perm_left.iter().map(|&i| g.left_shape[i]).collect();
            if left1.shape() != want_left.as_slice() {
                return Err(format!(
                    "transpose_read gave shape {:?}, expected {want_left:?}",
                    left1.shape()
                ));
            }
            emit(&[
                ("tag", q(&tag)),
                ("variant", q("relayout")),
                ("left_transpose_seconds", format!("{relayout_left:.6}")),
                ("left_elements", left_len.to_string()),
            ]);
            drop(left);
            drop(right);
            let reference = reference_tensor.as_slice::<Complex32>().map_err(err)?;
            let l1 = left1.as_slice::<Complex32>().map_err(err)?;
            let r1 = right1.as_slice::<Complex32>().map_err(err)?;
            for by_row in [false, true] {
                run_grouped(
                    &mut engine,
                    &g,
                    l1,
                    r1,
                    by_row,
                    0,
                    0,
                    Some(reference),
                    &tag,
                )?;
            }
        }
        "kernel" => {
            // One job of this step's (m, k) with `cols` columns on one thread, the
            // route each grouped job takes under the engine's batch policy, against
            // one plain pass over the job's left matrix.
            let left = random_c32(g.m * g.k, 1);
            let t = Instant::now();
            let copied = left.clone();
            let copy_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let sum: f32 = left.iter().map(|z| z.re).sum();
            let read_seconds = t.elapsed().as_secs_f64();
            drop(copied);
            emit(&[
                ("tag", q(&tag)),
                ("variant", q("kernel-reference")),
                ("left_elements", left.len().to_string()),
                ("one_thread_copy_seconds", format!("{copy_seconds:.6}")),
                ("one_thread_read_seconds", format!("{read_seconds:.6}")),
                ("checksum", format!("{sum:.3}")),
            ]);
            let one = Complex32::new(1.0, 0.0);
            let zero = Complex32::new(0.0, 0.0);
            for cols in [1usize, 2, 4, 8, 16, 26, 32, 64] {
                let rhs = random_c32(g.k * cols, 2);
                let jobs = [GroupedGemmJob::new(0, 0, 0, g.m, g.k, cols)];
                let plan = GroupedPlan::<Complex32>::new(&jobs, &PlanConfig::default())
                    .map_err(err)?;
                let mut out = Box::<[Complex32]>::new_uninit_slice(g.m * cols);
                let mut times = Vec::new();
                for _ in 0..(warm.max(1) + 1) {
                    let t = Instant::now();
                    plan.execute_uninit(
                        &mut cpueinsum::Exec::serial(),
                        one,
                        &left,
                        &rhs,
                        zero,
                        &mut out,
                    )
                    .map_err(err)?;
                    times.push(t.elapsed().as_secs_f64());
                }
                // The first run touches a fresh output; the rest reuse it.
                let warm_times = &times[1..];
                let best = warm_times.iter().copied().fold(f64::INFINITY, f64::min);
                emit(&[
                    ("tag", q(&tag)),
                    ("variant", q("kernel")),
                    ("m", g.m.to_string()),
                    ("k", g.k.to_string()),
                    ("cols", cols.to_string()),
                    ("first_seconds", format!("{:.6}", times[0])),
                    ("best_warm_seconds", format!("{best:.6}")),
                    ("madds_per_second", format!("{:.4e}", (g.m * g.k * cols) as f64 / best)),
                    ("left_reads_per_job", format!("{:.2}", best / read_seconds)),
                ]);
            }
        }
        other => return Err(format!("unknown mode `{other}`")),
    }
    Ok(())
}
