"""
process_rtmc_memory_time.py
===========================
پردازش تمام فایل‌های FASTA در پوشه RTMC با CEHO-MSA
و ثبت زمان پردازش + پیک حافظه مصرفی در یک فایل Excel.

نیازمندی‌ها:
    pip install psutil openpyxl

نحوه اجرا:
    python process_rtmc_memory_time.py
یا از طریق فایل Batch همراه.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import threading
from pathlib import Path

import psutil
from openpyxl import Workbook

# Import از فایل ceho_core.py (باید در همان پوشه باشد)
from ceho_core import (
    read_fasta,
    write_fasta,
    ceho_align,
    compute_sp,
    compute_cs,
    compute_gap_pct,
)

# ---------------------------------------------------------------------------
# تنظیمات پیش‌فرض (قابل تغییر با آرگومان خط فرمان)
# ---------------------------------------------------------------------------
RTMC_FOLDER = "RunTime_Memory"                # پوشه حاوی فایل‌های FASTA
OUTPUT_EXCEL = "memory_time_report.xlsx"
OUTPUT_ALN_DIR = "aligned_output"   # پوشه خروجی هم‌ترازی‌ها (اختیاری)

CEHO_KMER_K = 4
CEHO_EHO_ITERATIONS = 30
CEHO_EHO_POPULATION = 12
CEHO_NUM_CLANS = 3
CEHO_SEED = 42


# ---------------------------------------------------------------------------
# اندازه‌گیری پیک حافظه به کمک نمونه‌برداری دوره‌ای از RSS
# ---------------------------------------------------------------------------
class MemoryMonitor:
    """یک نخ جداگانه که در طول اجرا، حافظه مصرفی فرآیند را نمونه‌برداری می‌کند
    و بیشترین مقدار مشاهده‌شده را نگه می‌دارد."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process()

    def _monitor(self):
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss  # bytes
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """توقف نمونه‌برداری و بازگرداندن پیک بر حسب مگابایت"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_rss / (1024 * 1024)  # MB


# ---------------------------------------------------------------------------
# پردازش یک فایل FASTA
# ---------------------------------------------------------------------------
def process_fasta(fasta_path: str, output_aln_dir: str | None) -> dict:
    """اجرای CEHO-MSA روی یک فایل و برگرداندن نتایج اندازه‌گیری."""
    base = Path(fasta_path).stem
    print(f"\n--- پردازش {base} ---")

    # خواندن توالی‌ها
    seqs = read_fasta(fasta_path)
    n_seqs = len(seqs)
    print(f"  تعداد توالی‌ها: {n_seqs}")

    # راه‌اندازی مانیتور حافظه و زمان‌سنج
    monitor = MemoryMonitor(interval=0.05)
    monitor.start()
    start_time = time.perf_counter()

    try:
        aligned = ceho_align(
            seqs,
            k=CEHO_KMER_K,
            eho_iterations=CEHO_EHO_ITERATIONS,
            eho_population=CEHO_EHO_POPULATION,
            num_clans=CEHO_NUM_CLANS,
            seed=CEHO_SEED,
        )
    except Exception as e:
        monitor.stop()
        elapsed = time.perf_counter() - start_time
        print(f"  خطا در پردازش: {e}")
        return {
            "file": base,
            "num_sequences": n_seqs,
            "alignment_length": 0,
            "SP_score": None,
            "CS_score": None,
            "Gap_percent": None,
            "time_sec": round(elapsed, 3),
            "peak_memory_MB": round(monitor.peak_rss / (1024 * 1024), 2),
            "status": "FAILED",
        }

    elapsed = time.perf_counter() - start_time
    peak_mb = monitor.stop()

    # محاسبه معیارها
    sp = compute_sp(aligned)
    cs = round(compute_cs(aligned), 4)
    gap = round(compute_gap_pct(aligned), 2)
    aln_len = len(next(iter(aligned.values())))

    # ذخیره هم‌ترازی در صورت نیاز
    if output_aln_dir:
        out_path = os.path.join(output_aln_dir, f"{base}_CEHO.fasta")
        write_fasta(aligned, out_path)

    print(f"  طول هم‌ترازی: {aln_len}")
    print(f"  SP={sp}, CS={cs}, Gap%={gap}")
    print(f"  زمان: {elapsed:.2f} ثانیه")
    print(f"  پیک حافظه: {peak_mb:.2f} MB")

    return {
        "file": base,
        "num_sequences": n_seqs,
        "alignment_length": aln_len,
        "SP_score": sp,
        "CS_score": cs,
        "Gap_percent": gap,
        "time_sec": round(elapsed, 3),
        "peak_memory_MB": round(peak_mb, 2),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# تابع اصلی
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="پردازش فایل‌های FASTA در RTMC و ثبت زمان/حافظه در Excel"
    )
    parser.add_argument("--input", default=RTMC_FOLDER, help="پوشه حاوی فایل‌های FASTA")
    parser.add_argument("--output-excel", default=OUTPUT_EXCEL, help="مسیر فایل Excel خروجی")
    parser.add_argument("--output-aln", default=OUTPUT_ALN_DIR,
                        help="پوشه ذخیره هم‌ترازی‌ها (برای غیرفعال کردن، مقدار خالی بدهید)")
    args = parser.parse_args()

    # ساخت پوشه خروجی هم‌ترازی‌ها در صورت نیاز
    if args.output_aln:
        os.makedirs(args.output_aln, exist_ok=True)

    # پیدا کردن فایل‌های FASTA
    patterns = ["*.fasta", "*.fa", "*.fna"]
    fasta_files = []
    for pat in patterns:
        fasta_files.extend(glob.glob(os.path.join(args.input, pat)))
    fasta_files = sorted(set(fasta_files))

    if not fasta_files:
        print(f"هیچ فایل FASTA در پوشه '{args.input}' یافت نشد.")
        return

    print(f"تعداد فایل‌های یافت‌شده: {len(fasta_files)}")

    results = []
    for fpath in fasta_files:
        res = process_fasta(fpath, args.output_aln)
        results.append(res)

    # نوشتن در Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "CEHO-MSA Performance"

    # سربرگ‌ها
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
    print(f"\nگزارش Excel در '{args.output_excel}' ذخیره شد.")


if __name__ == "__main__":
    main()
