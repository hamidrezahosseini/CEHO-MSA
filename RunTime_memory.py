"""
process_magus_memory_time.py
=============================
پردازش تمام فایل‌های FASTA در پوشه RunTime_Memory با MAGUS
و ثبت زمان اجرا + پیک حافظه مصرفی (مجموع RSS کل فرآیندها) در فایل Excel.

پیش‌نیازها:
    pip install psutil openpyxl

اجرا:
    python process_magus_memory_time.py
یا با آرگومان (اختیاری):
    python process_magus_memory_time.py --input RunTime_Memory --output-excel magus_report.xlsx
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import psutil
from openpyxl import Workbook

# برای محاسبه معیارهای هم‌ترازی (SP/CS/Gap) از توابع ceho_core استفاده می‌کنیم
# (این فایل باید در همان پوشه باشد)
from ceho_core import (
    read_fasta,
    compute_sp,
    compute_cs,
    compute_gap_pct,
)

# ---------------------------------------------------------------------------
# CONFIGURATION (مطابق اسکریپت MAGUS شما)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "RunTime_Memory")   # پوشه ورودی پیش‌فرض
OUTPUT_DIR = os.path.join(BASE_DIR, "output_magus")    # خروجی هم‌ترازی‌ها
WORKDIR_ROOT = os.path.join(BASE_DIR, "magus_work")    # پوشه فایل‌های میانی

MAGUS_EXE = (
    r"C:\Users\HamidrezaHosseini\AppData\Local"
    r"\Programs\Python\Python39\Scripts\magus.exe"
)

MAFFT_PATH = (
    r"C:\Program Files (x86)\ClustalW2\mafft-win\mafft.bat"
)

# تنظیمات MAGUS (همان مقادیر استفاده‌شده در magus_batch.py)
DATATYPE = None
NUM_PROCS = None
MAFFT_RUNS = 1
MAFFT_SIZE = None
GRAPH_BUILD_METHOD = "mafft"
GUIDETREE = "parttree"
GRAPH_CLUSTER_METHOD = "none"
OVERWRITE = False

FASTA_EXTENSIONS = (".fasta", ".fa", ".fas", ".fna")

# فاصله نمونه‌برداری حافظه (ثانیه)
SAMPLE_INTERVAL = 0.1


# ---------------------------------------------------------------------------
# اندازه‌گیری پیک حافظه کل درخت فرآیندها (MAGUS و فرزندان)
# ---------------------------------------------------------------------------
class ProcessTreeMemoryMonitor:
    """نمونه‌برداری دوره‌ای از مجموع RSS فرآیند اصلی و تمام فرزندان آن."""

    def __init__(self, pid: int, interval: float = SAMPLE_INTERVAL):
        self.pid = pid
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _get_tree_rss(self) -> int:
        """مجموع RSS فرآیند اصلی و تمام فرزندان (به بایت)."""
        try:
            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
            total = 0
            for p in procs:
                try:
                    total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except psutil.NoSuchProcess:
            return 0

    def _monitor(self):
        while not self._stop.is_set():
            try:
                rss = self._get_tree_rss()
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """توقف و بازگرداندن پیک بر حسب مگابایت."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# ساخت دستور MAGUS
# ---------------------------------------------------------------------------
def build_command(input_file: str, output_file: str, workdir: str) -> list:
    command = [
        MAGUS_EXE,
        "-d", workdir,
        "-i", input_file,
        "-o", output_file,
    ]

    if DATATYPE is not None:
        command.extend(["--datatype", DATATYPE])
    if NUM_PROCS is not None:
        command.extend(["-np", str(NUM_PROCS)])
    if MAFFT_RUNS is not None:
        command.extend(["-r", str(MAFFT_RUNS)])
    if MAFFT_SIZE is not None:
        command.extend(["-m", str(MAFFT_SIZE)])
    if GRAPH_BUILD_METHOD is not None:
        command.extend(["--graphbuildmethod", GRAPH_BUILD_METHOD])
    if GUIDETREE is not None:
        command.extend(["-t", GUIDETREE])
    if GRAPH_CLUSTER_METHOD is not None:
        command.extend(["--graphclustermethod", GRAPH_CLUSTER_METHOD])

    return command


# ---------------------------------------------------------------------------
# اجرای MAGUS با اندازه‌گیری زمان و حافظه
# ---------------------------------------------------------------------------
def run_magus_measured(input_file: str, output_file: str, workdir: str):
    """اجرای MAGUS با Popen و مانیتورینگ حافظه، بازگرداندن نتایج."""
    command = build_command(input_file, output_file, workdir)
    print("MAGUS command:")
    print(" ".join('"' + x + '"' if " " in x else x for x in command))

    # ساخت workdir اگر وجود ندارد
    os.makedirs(workdir, exist_ok=True)

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"خطا در شروع MAGUS: {e}")
        return False, elapsed, 0.0, "", str(e)

    # شروع مانیتور حافظه
    monitor = ProcessTreeMemoryMonitor(process.pid, interval=SAMPLE_INTERVAL)
    monitor.start()

    # انتظار برای پایان فرآیند
    try:
        stdout_text, stderr_text = process.communicate()
    except Exception as e:
        process.kill()
        stdout_text, stderr_text = process.communicate()
        elapsed = time.perf_counter() - start_time
        peak_mb = monitor.stop()
        print(f"خطا در اجرای MAGUS: {e}")
        return False, elapsed, peak_mb, stdout_text, stderr_text

    elapsed = time.perf_counter() - start_time
    peak_mb = monitor.stop()

    # بررسی خروجی
    if process.returncode != 0:
        print(f"returncode = {process.returncode}")
        if stderr_text:
            print("stderr:", stderr_text[-500:])
        # چاپ log در صورت وجود
        _print_workdir_log(workdir)
        return False, elapsed, peak_mb, stdout_text, stderr_text

    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        print("فایل خروجی ساخته نشد یا خالی است.")
        _print_workdir_log(workdir)
        return False, elapsed, peak_mb, stdout_text, stderr_text

    return True, elapsed, peak_mb, stdout_text, stderr_text


def _print_workdir_log(workdir: str, max_lines=30):
    """چاپ محتوای فایل‌های log در workdir برای دیباگ."""
    candidates = glob.glob(os.path.join(workdir, "*log*"))
    if not candidates:
        print("(هیچ فایل log ای یافت نشد)")
        return
    for log_path in candidates:
        print(f"\n--- محتوای {log_path} (آخرین {max_lines} خط) ---")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            for line in lines[-max_lines:]:
                print(line.rstrip())
        except Exception as e:
            print(f"(خواندن log ممکن نشد: {e})")


# ---------------------------------------------------------------------------
# پردازش یک فایل
# ---------------------------------------------------------------------------
def process_file(filename: str, index: int, total: int, output_aln_dir: str):
    base = Path(filename).stem
    print(f"\n{'='*70}\nFILE {index}/{total}: {base}\n{'='*70}")

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(output_aln_dir, filename)
    workdir = os.path.join(WORKDIR_ROOT, base)

    # بررسی skip
    if not OVERWRITE and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        print("STATUS: SKIP (already processed)")
        return {
            "file": base,
            "status": "SKIP",
            "num_sequences": 0,
            "alignment_length": 0,
            "SP_score": None,
            "CS_score": None,
            "Gap_percent": None,
            "time_sec": 0.0,
            "peak_memory_MB": 0.0,
        }

    print(f"INPUT : {input_path}")
    print(f"OUTPUT: {output_path}")
    print(f"WORKDIR: {workdir}")

    # شمارش توالی‌ها قبل از اجرا
    try:
        seqs = read_fasta(input_path)
        n_seqs = len(seqs)
    except Exception:
        n_seqs = 0

    print("STATUS: RUNNING MAGUS...")
    success, elapsed, peak_mb, stdout_text, stderr_text = run_magus_measured(
        input_path, output_path, workdir
    )

    if not success:
        print(f"STATUS: FAILED after {elapsed:.2f}s, peak mem {peak_mb:.2f} MB")
        # حذف فایل خروجی ناقص
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
                print("Incomplete output removed.")
            except Exception:
                pass
        return {
            "file": base,
            "status": "FAILED",
            "num_sequences": n_seqs,
            "alignment_length": 0,
            "SP_score": None,
            "CS_score": None,
            "Gap_percent": None,
            "time_sec": round(elapsed, 3),
            "peak_memory_MB": round(peak_mb, 2),
        }

    # موفقیت: خواندن خروجی و محاسبه معیارها
    try:
        aligned = read_fasta(output_path)
        sp = compute_sp(aligned)
        cs = round(compute_cs(aligned), 4)
        gap = round(compute_gap_pct(aligned), 2)
        aln_len = len(next(iter(aligned.values())))
        n_seqs_out = len(aligned)
    except Exception as e:
        print(f"خطا در خواندن خروجی: {e}")
        sp, cs, gap, aln_len, n_seqs_out = None, None, None, 0, 0

    print(f"STATUS: SUCCESS in {elapsed:.2f}s, peak mem {peak_mb:.2f} MB")
    if sp is not None:
        print(f"  #Seq={n_seqs_out}, Length={aln_len}, SP={sp}, CS={cs}, Gap%={gap}")

    return {
        "file": base,
        "status": "SUCCESS",
        "num_sequences": n_seqs_out,
        "alignment_length": aln_len,
        "SP_score": sp,
        "CS_score": cs,
        "Gap_percent": gap,
        "time_sec": round(elapsed, 3),
        "peak_memory_MB": round(peak_mb, 2),
    }


# ---------------------------------------------------------------------------
# تابع اصلی
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="پردازش فایل‌های FASTA در RunTime_Memory با MAGUS و ثبت زمان/حافظه در Excel"
    )
    parser.add_argument("--input", default=INPUT_DIR, help="پوشه حاوی فایل‌های FASTA (پیش‌فرض: RunTime_Memory)")
    parser.add_argument("--output-aln", default=OUTPUT_DIR, help="پوشه ذخیره هم‌ترازی‌های MAGUS")
    parser.add_argument("--output-excel", default="magus_memory_time_report.xlsx",
                        help="مسیر فایل Excel خروجی")
    args = parser.parse_args()

    # به‌روزرسانی مسیرهای سراسری
    INPUT_DIR, OUTPUT_DIR
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output_aln

    # ساخت پوشه‌های لازم
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORKDIR_ROOT, exist_ok=True)

    # بررسی وجود MAGUS
    if not os.path.isfile(MAGUS_EXE):
        print(f"ERROR: MAGUS executable not found at:\n{MAGUS_EXE}")
        sys.exit(1)

    # پیدا کردن فایل‌های FASTA
    fasta_files = []
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(FASTA_EXTENSIONS) and os.path.isfile(os.path.join(INPUT_DIR, fname)):
            fasta_files.append(fname)
    fasta_files.sort()

    if not fasta_files:
        print(f"No FASTA files found in '{INPUT_DIR}'.")
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA file(s). Starting processing...\n")

    results = []
    total_start = time.time()

    for i, fname in enumerate(fasta_files, start=1):
        res = process_file(fname, i, len(fasta_files), OUTPUT_DIR)
        results.append(res)

    total_elapsed = time.time() - total_start

    # نوشتن Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "MAGUS Performance"

    headers = [
        "File", "Status", "#Sequences", "Alignment_Length",
        "SP_Score", "CS_Score", "Gap_Percent",
        "Time_sec", "Peak_Memory_MB"
    ]
    ws.append(headers)

    for r in results:
        ws.append([
            r["file"],
            r["status"],
            r["num_sequences"],
            r["alignment_length"],
            r["SP_score"],
            r["CS_score"],
            r["Gap_percent"],
            r["time_sec"],
            r["peak_memory_MB"],
        ])

    # تنظیم عرض ستون‌ها
    for col in ws.columns:
        max_len = max(len(str(cell.value)) for cell in col if cell.value is not None)
        ws.column_dimensions[col[0].column_letter].width = max_len + 2

    wb.save(args.output_excel)

    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    print(f"Total files : {len(results)}")
    print(f"SUCCESS     : {success}")
    print(f"FAILED      : {failed}")
    print(f"SKIPPED     : {skipped}")
    print(f"Total time  : {total_elapsed:.2f} seconds")
    print(f"\nExcel report saved to: {args.output_excel}")


if __name__ == "__main__":
    main()
