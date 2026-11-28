//! Driver for tenferro-einsum's `step172-exp` measurement (tensor4all/tenferro-rs#2756).
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = tenferro_einsum::step172_exp_run(&args) {
        eprintln!("step172: {e}");
        std::process::exit(1);
    }
}
