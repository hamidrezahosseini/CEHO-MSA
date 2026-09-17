"""
run_msa_comparison.py
======================
Batch-runs three multiple sequence alignment methods over every FASTA file
found in an input folder, and writes the resulting alignments plus an
SP-score / Column-Score / Gap-percentage comparison table to an output
folder:

  1. CEHO-MSA  -- the clustering + Elephant Herding Optimization method
                  implemented in ceho_core.py (this submission's new method).
  2. ClustalW2 -- run via the local clustalw2.exe binary.
  3. MAFFT     -- run via the local mafft.bat wrapper.

Usage
-----
Just edit the CONFIGURATION block below (paths already filled in for your
machine) and run:

    python run_msa_comparison.py

Or override at the command line:

    python run_msa_comparison.py --input C:/path/to/input --output C:/path/to/output

Input folder: any *.fasta / *.fa files, each containing >= 1 un-aligned
sequence in standard FASTA format.

Output folder: for every input file "<name>.fasta" this script writes
    <name>_CEHO.fasta
    <name>_ClustalW.fasta
    <name>_MAFFT.fasta
and a single combined report:
    comparison_summary.csv
    comparison_report.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time

from ceho_core import (
    read_fasta,
    write_fasta,
    is_dna,
    ceho_align,
    compute_sp,
    compute_cs,
    compute_gap_pct,
)

# ---------------------------------------------------------------------------
# CONFIGURATION -- edit these for your machine, or override via CLI flags
# ---------------------------------------------------------------------------
MAFFT_PATH = r"C:\Program Files (x86)\ClustalW2\mafft-win\mafft.bat"
CLUSTALW_PATH = r"C:\Program Files (x86)\ClustalW2\clustalw2.exe"

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# CEHO-MSA hyper-parameters (see ceho_core.ceho_align for details)
CEHO_KMER_K = 4
CEHO_EHO_ITERATIONS = 30
CEHO_EHO_POPULATION = 12
CEHO_NUM_CLANS = 3
CEHO_SEED = 42


# ---------------------------------------------------------------------------
# External tool wrappers
# ---------------------------------------------------------------------------
def run_clustalw(fasta_path: str, out_path: str, clustalw_path: str) -> "dict[str, str]":
    seqs = read_fasta(fasta_path)
    seqtype = "DNA" if is_dna(seqs) else "PROTEIN"
    cmd = [
        clustalw_path,
        f"-INFILE={fasta_path}",
        f"-OUTFILE={out_path}",
        "-OUTPUT=FASTA",
        f"-TYPE={seqtype}",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"ClustalW2 failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='ignore')[:500]}"
        )
    return read_fasta(out_path)


def run_mafft(fasta_path: str, out_path: str, mafft_path: str) -> "dict[str, str]":
    with open(out_path, "w") as out_fh:
        result = subprocess.run(
            f'"{mafft_path}" "{fasta_path}"',
            stdout=out_fh,
            stderr=subprocess.PIPE,
            shell=True,
        )
    if result.returncode != 0 or os.path.getsize(out_path) == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(
            f"MAFFT failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='ignore')[:500]}"
        )
    return read_fasta(out_path)


def run_ceho(fasta_path: str, out_path: str) -> "dict[str, str]":
    seqs = read_fasta(fasta_path)
    aligned = ceho_align(
        seqs,
        k=CEHO_KMER_K,
        eho_iterations=CEHO_EHO_ITERATIONS,
        eho_population=CEHO_EHO_POPULATION,
        num_clans=CEHO_NUM_CLANS,
        seed=CEHO_SEED,
    )
    write_fasta(aligned, out_path)
    return aligned


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(aligned: "dict[str, str]") -> "dict[str, float]":
    seqs = list(aligned.values())
    return {
        "num_sequences": len(seqs),
        "alignment_length": len(seqs[0]) if seqs else 0,
        "SP_score": compute_sp(aligned),
        "CS_score": round(compute_cs(aligned), 4),
        "Gap_percent": round(compute_gap_pct(aligned), 2),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main() -> None:
    global CEHO_EHO_ITERATIONS, CEHO_EHO_POPULATION, CEHO_NUM_CLANS
    parser = argparse.ArgumentParser(description="Compare CEHO-MSA vs ClustalW2 vs MAFFT")
    parser.add_argument("--input", default=INPUT_DIR, help="Folder containing input FASTA files")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder to write alignments + report")
    parser.add_argument("--mafft", default=MAFFT_PATH, help="Path to mafft.bat")
    parser.add_argument("--clustalw", default=CLUSTALW_PATH, help="Path to clustalw2.exe")
    parser.add_argument("--skip-clustalw", action="store_true", help="Skip the ClustalW2 run")
    parser.add_argument("--skip-mafft", action="store_true", help="Skip the MAFFT run")
    parser.add_argument("--eho-iterations", type=int, default=CEHO_EHO_ITERATIONS)
    parser.add_argument("--eho-population", type=int, default=CEHO_EHO_POPULATION)
    parser.add_argument("--num-clans", type=int, default=CEHO_NUM_CLANS)
    args = parser.parse_args()

    CEHO_EHO_ITERATIONS = args.eho_iterations
    CEHO_EHO_POPULATION = args.eho_population
    CEHO_NUM_CLANS = args.num_clans

    os.makedirs(args.output, exist_ok=True)

    fasta_files = sorted(
        set(glob.glob(os.path.join(args.input, "*.fasta")))
        | set(glob.glob(os.path.join(args.input, "*.fa")))
        | set(glob.glob(os.path.join(args.input, "*.fna")))
    )

    if not fasta_files:
        print(f"No FASTA files (*.fasta/*.fa/*.fna) found in '{args.input}'.")
        sys.exit(1)

    summary_rows = []

    for fasta_path in fasta_files:
        base = os.path.splitext(os.path.basename(fasta_path))[0]
        print(f"\n=== {base} ===")

        n_seqs = len(read_fasta(fasta_path))
        print(f"  {n_seqs} input sequences")

        # ---- CEHO-MSA -------------------------------------------------
        try:
            out_path = os.path.join(args.output, f"{base}_CEHO.fasta")
            t0 = time.time()
            aligned = run_ceho(fasta_path, out_path)
            elapsed = time.time() - t0
            metrics = evaluate(aligned)
            metrics.update(dataset=base, method="CEHO-MSA", runtime_s=round(elapsed, 3))
            summary_rows.append(metrics)
            print(f"  CEHO-MSA   : SP={metrics['SP_score']}  CS={metrics['CS_score']}  "
                  f"Gap%={metrics['Gap_percent']}  ({elapsed:.2f}s)")
        except Exception as exc:
            print(f"  CEHO-MSA   : FAILED -- {exc}")

        # ---- ClustalW2 --------------------------------------------------
        if not args.skip_clustalw:
            try:
                out_path = os.path.join(args.output, f"{base}_ClustalW.fasta")
                t0 = time.time()
                aligned = run_clustalw(fasta_path, out_path, args.clustalw)
                elapsed = time.time() - t0
                metrics = evaluate(aligned)
                metrics.update(dataset=base, method="ClustalW2", runtime_s=round(elapsed, 3))
                summary_rows.append(metrics)
                print(f"  ClustalW2  : SP={metrics['SP_score']}  CS={metrics['CS_score']}  "
                      f"Gap%={metrics['Gap_percent']}  ({elapsed:.2f}s)")
            except Exception as exc:
                print(f"  ClustalW2  : FAILED -- {exc}")

        # ---- MAFFT --------------------------------------------------------
        if not args.skip_mafft:
            try:
                out_path = os.path.join(args.output, f"{base}_MAFFT.fasta")
                t0 = time.time()
                aligned = run_mafft(fasta_path, out_path, args.mafft)
                elapsed = time.time() - t0
                metrics = evaluate(aligned)
                metrics.update(dataset=base, method="MAFFT", runtime_s=round(elapsed, 3))
                summary_rows.append(metrics)
                print(f"  MAFFT      : SP={metrics['SP_score']}  CS={metrics['CS_score']}  "
                      f"Gap%={metrics['Gap_percent']}  ({elapsed:.2f}s)")
            except Exception as exc:
                print(f"  MAFFT      : FAILED -- {exc}")

    # ---- write CSV summary -------------------------------------------------
    csv_path = os.path.join(args.output, "comparison_summary.csv")
    fieldnames = ["dataset", "method", "num_sequences", "alignment_length",
                  "SP_score", "CS_score", "Gap_percent", "runtime_s"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # ---- write a human-readable text report --------------------------------
    txt_path = os.path.join(args.output, "comparison_report.txt")
    with open(txt_path, "w") as fh:
        fh.write("CEHO-MSA vs ClustalW2 vs MAFFT -- comparison report\n")
        fh.write("=" * 60 + "\n\n")
        by_dataset: "dict[str, list]" = {}
        for row in summary_rows:
            by_dataset.setdefault(row["dataset"], []).append(row)
        for dataset, rows in by_dataset.items():
            fh.write(f"Dataset: {dataset}\n")
            fh.write(f"{'Method':<12}{'#Seq':>6}{'Length':>8}{'SP-score':>12}"
                      f"{'CS':>8}{'Gap%':>8}{'Time(s)':>10}\n")
            for r in rows:
                fh.write(f"{r['method']:<12}{r['num_sequences']:>6}{r['alignment_length']:>8}"
                          f"{r['SP_score']:>12}{r['CS_score']:>8}{r['Gap_percent']:>8}"
                          f"{r['runtime_s']:>10}\n")
            best = max(rows, key=lambda r: r["SP_score"])
            fh.write(f"  -> Highest SP-score: {best['method']} ({best['SP_score']})\n\n")

    print(f"\nSummary written to:\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
