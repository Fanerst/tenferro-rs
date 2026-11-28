"""Write n30m14 as the plain-text case the step-172 harness reads.

Same semantics as review-2739/scripts/gen_cases.py's "n30m14" case: every output label
is sampled, sample values are the bitstrings' bits (column-major [n, labels]), every
extent is 2, and the path is network.json's SSA path.

Usage: python3 make_case.py DATA_DIR OUT.txt
"""
import json
import sys


def main():
    d, out = sys.argv[1], sys.argv[2]
    with open(f"{d}/network.json") as f:
        ex = json.load(f)
    inputs, output, path, bits = ex["inputs"], ex["output"], ex["path"], ex["bitstrings"]
    n, L = len(bits), len(output)
    values = [0] * (n * L)
    for s, b in enumerate(bits):
        assert len(b) == L
        for j, c in enumerate(b):
            values[s + n * j] = 1 if c == "1" else 0
    lines = [f"inputs {len(inputs)}"]
    lines += [" ".join(map(str, [len(ls)] + ls)) for ls in inputs]
    lines.append("output " + " ".join(map(str, [len(output)] + output)))
    lines.append(f"path {len(path)}")
    lines += [f"{a} {b}" for a, b in path]
    lines.append(f"samples {n} {L}")
    lines.append(" ".join(map(str, output)))
    lines.append(" ".join(map(str, values)))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(inputs)} inputs, {len(path)} steps, {n} samples of {L} labels")


if __name__ == "__main__":
    main()
