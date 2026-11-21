"""One visitor's demo state. Photos are decoded in memory."""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from threading import BoundedSemaphore

import numpy as np

from demo.interface_text import TEXT
from idetectface import recognition
from idetectface.clock import bangladesh_time
from idetectface.images import decode_image

# The shared CPU models operate on one bounded image at a time.
_inference_slot = BoundedSemaphore(1)


class PhotoError(ValueError):
    def __init__(self, reason: str):
        super().__init__(TEXT["English"][reason])
        self.reason = reason


def encode_photo(raw: bytes):
    if len(raw) > 4 * 1024 * 1024:
        raise PhotoError("large")
    if not _inference_slot.acquire(timeout=2):
        raise PhotoError("busy")
    try:
        try:
            rgb, digest = decode_image(raw, max_side=640)
        except (ValueError, OSError) as error:
            raise PhotoError("invalid") from error
        boxes = recognition.face_locations(rgb)
        if len(boxes) != 1:
            raise PhotoError("many" if boxes else "none")
        (vector,) = recognition.face_encodings(rgb, boxes)
        return vector, digest
    finally:
        _inference_slot.release()


@dataclass
class Visit:
    reference: np.ndarray
    enrollment_digest: bytes
    entries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_utc_session(cls, previous):
        """Keep the earliest surviving check-in on each Bangladesh date after an app update."""
        visit = cls(previous.reference, previous.enrollment_digest)
        for day, at in sorted(previous.entries.items()):
            stamp = bangladesh_time(datetime.fromisoformat(f"{day}T{at}+00:00"))
            visit.entries.setdefault(stamp.date().isoformat(), stamp.strftime("%H:%M:%S"))
        return visit

    def check_in(self, vector, digest, now=None):
        if digest == self.enrollment_digest:
            raise PhotoError("same")
        distance = float(recognition.face_distance(self.reference[None, :], vector)[0])
        if not distance < 0.5:
            return "Unknown", distance
        stamp = bangladesh_time(now)
        day = stamp.date().isoformat()
        if day in self.entries:
            return "Already checked in today", distance
        self.entries[day] = stamp.strftime("%H:%M:%S")
        return "Checked in", distance

    def csv_text(self):
        buffer = StringIO(newline="")
        rows = csv.writer(buffer)
        rows.writerow(("Name", "Date", "Time"))
        rows.writerows(("visitor", day, at) for day, at in self.entries.items())
        return buffer.getvalue()
