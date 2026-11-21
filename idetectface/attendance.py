"""An append-only CSV register shared by cooperating processes on one machine."""

import csv
import locale
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from filelock import FileLock

from .identity import identity_key


class Record(NamedTuple):
    name: str
    day: str
    at: str


class AttendanceLog:
    header = ("Name", "Date", "Time")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=5)
        self._snapshot = None
        self._seen = set()
        self._encoding = "utf-8"

    def _version(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _rows(self, encoding: str) -> Iterator[Record]:
        if not self.path.exists():
            return
        with self.path.open(encoding=encoding, newline="") as stream:
            rows = csv.reader(stream, strict=True)
            if tuple(next(rows, ())) != self.header:
                raise ValueError(f"{self.path} must start with Name,Date,Time.")
            for row in rows:
                try:
                    record = Record(*row)
                    if not record.name.strip():
                        raise ValueError("Empty name")
                    if date.fromisoformat(record.day).isoformat() != record.day:
                        raise ValueError("Invalid date")
                    if time.fromisoformat(record.at).strftime("%H:%M:%S") != record.at:
                        raise ValueError("Invalid time")
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid attendance record at line {rows.line_num} in {self.path}."
                    ) from error
                yield record

    def _select(self, day: Optional[str], *, names_only: bool = False):
        def collect(encoding):
            rows = (r for r in self._rows(encoding) if day is None or r.day == day)
            return {identity_key(r.name) for r in rows} if names_only else list(rows)

        try:
            result = collect("utf-8-sig")
            self._encoding = "utf-8"
        except UnicodeDecodeError:
            # The original app wrote using the system's default text encoding.
            self._encoding = locale.getpreferredencoding(False)
            result = collect(self._encoding)
        return result

    def records(self, day: Optional[date] = None) -> list:
        with self.lock:
            return self._select(day.isoformat() if day else None)

    def mark(self, name: str, now: Optional[datetime] = None) -> bool:
        """Return True only for the first accepted write for an identity and local day."""
        if not name.strip() or name.lstrip()[0] in "=+-@" or any(ord(c) < 32 for c in name):
            raise ValueError("Unsafe or empty attendance name.")
        now = now or datetime.now()
        day = now.date().isoformat()
        key = identity_key(name)
        with self.lock:
            snapshot = (day, self._version())
            if self._snapshot != snapshot:
                self._seen = self._select(day, names_only=True)
                self._snapshot = snapshot
            if key in self._seen:
                return False
            row = Record(name, day, now.strftime("%H:%M:%S"))
            self._append(row)
            self._seen.add(key)
            self._snapshot = (day, self._version())
            return True

    def _append(self, row: Record) -> None:
        # Some valid CSV editors omit the final line break.
        exists = self.path.exists()
        needs_newline = False
        if exists:
            with self.path.open("rb") as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) not in (b"\r", b"\n")
        with self.path.open("a", encoding=self._encoding, newline="") as stream:
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(self.header)
            elif needs_newline:
                stream.write("\r\n")
            writer.writerow(row)
