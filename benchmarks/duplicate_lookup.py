"""Time a repeated sighting with the matching entry at the end of a CSV."""

import csv
import platform
import statistics
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import repeat

from idetectface.attendance import AttendanceLog

NOW = datetime(2025, 3, 9, 9)
DAY = NOW.date().isoformat()
BATCH = 50
REPEATS = 7


def baseline(path):
    # Same duplicate scan and early exit as the original mark_attendance.
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        for row in rows:
            if len(row) >= 2 and row[0] == "demo-last" and row[1] == DAY:
                return False
    raise AssertionError("Missing benchmark identity")


def micros_per_check(operation):
    return statistics.median(repeat(operation, number=BATCH, repeat=REPEATS)) * 1e6 / BATCH


def main():
    print(f"{platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"Median of {REPEATS} batches x {BATCH} checks; microseconds per check")
    print("rows     original scan     indexed lookup")
    with TemporaryDirectory() as folder:
        for size in (100, 1_000, 10_000):
            path = Path(folder) / f"{size}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(AttendanceLog.header)
                for i in range(size - 1):
                    writer.writerow((f"demo-{i}", "2025-03-08", "09:00:00"))
                writer.writerow(("demo-last", DAY, "09:00:00"))
            log = AttendanceLog(path)

            def indexed():
                return log.mark("demo-last", NOW)

            assert baseline(path) is False and indexed() is False
            scan = micros_per_check(lambda: baseline(path))
            cached = micros_per_check(indexed)
            print(f"{size:5} {scan:17.1f} {cached:18.1f}")


if __name__ == "__main__":
    main()
