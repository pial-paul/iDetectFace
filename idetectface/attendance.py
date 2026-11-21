"""An append-only CSV register shared by cooperating processes on one machine."""

import csv
import locale
import os
import stat
from datetime import date, datetime, time
from io import StringIO
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from filelock import FileLock

from .clock import bangladesh_time
from .identity import identity_key, validate_label


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
        self._last_present = None
        self._encoding = "utf-8"

    def _version(self):
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("The attendance register must be a regular file.")
        return info.st_ino, info.st_size, info.st_mtime_ns

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
                    validate_label(record.name)
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
            self._version()
            return self._select(day.isoformat() if day else None)

    def mark(self, name: str, now: Optional[datetime] = None) -> bool:
        """Return True only for the first accepted write for an identity on a Bangladesh date."""
        validate_label(name)
        now = bangladesh_time(now)
        day = now.date().isoformat()
        key = identity_key(name)
        # A repeat of the last committed sighting is read-only. Any date or
        # file-version change sends the caller through the locked check below.
        if self._last_present == (key, day, self._version()):
            return False
        with self.lock:
            snapshot = (day, self._version())
            if self._snapshot != snapshot:
                self._seen = self._select(day, names_only=True)
                self._snapshot = snapshot
            if key in self._seen:
                self._last_present = (key, day, snapshot[1])
                return False
            row = Record(name, day, now.strftime("%H:%M:%S"))
            self._append(row)
            self._seen.add(key)
            self._snapshot = (day, self._version())
            self._last_present = (key, day, self._snapshot[1])
            return True

    def _append(self, row: Record) -> None:
        exists = self.path.exists()
        buffer = StringIO(newline="")
        writer = csv.writer(buffer)
        if not exists:
            writer.writerow(self.header)
        writer.writerow(row)
        # Finish encoding before opening a legacy register for modification.
        payload = buffer.getvalue().encode(self._encoding)
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if not exists:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(self.path, flags, 0o600)
        try:
            with os.fdopen(fd, "r+b", buffering=0) as stream:
                size = stream.seek(0, os.SEEK_END)
                if size:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) not in (b"\r", b"\n"):
                        payload = b"\r\n" + payload
                try:
                    if stream.write(payload) != len(payload):
                        raise OSError("Incomplete attendance write.")
                    os.fsync(fd)
                except BaseException:
                    stream.truncate(size)
                    raise
        except BaseException:
            if not exists:
                self.path.unlink(missing_ok=True)
            raise
