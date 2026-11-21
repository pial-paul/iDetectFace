"""One visitor's demo state. Photos are decoded in memory."""

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from threading import BoundedSemaphore

import numpy as np

from idetectface import recognition
from idetectface.images import decode_image

# The shared CPU models operate on one bounded image at a time.
_inference_slot = BoundedSemaphore(1)


def encode_photo(raw: bytes):
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Choose a JPEG or PNG smaller than 4 MiB.")
    if not _inference_slot.acquire(timeout=2):
        raise ValueError("Another photo is being processed. Try again in a moment.")
    try:
        try:
            rgb, digest = decode_image(raw, max_side=640)
        except (ValueError, OSError) as error:
            raise ValueError("Choose a readable JPEG or PNG of up to 16 megapixels.") from error
        boxes = recognition.face_locations(rgb)
        if len(boxes) != 1:
            raise ValueError(f"Detected {len(boxes)} faces. Try a well-lit photo of just yourself.")
        (vector,) = recognition.face_encodings(rgb, boxes)
        return vector, digest
    finally:
        _inference_slot.release()


@dataclass
class Visit:
    reference: np.ndarray
    enrollment_digest: bytes
    entries: dict[str, str] = field(default_factory=dict)

    def check_in(self, vector, digest, now=None):
        if digest == self.enrollment_digest:
            raise ValueError("Use a second photo for the check-in.")
        distance = float(recognition.face_distance(self.reference[None, :], vector)[0])
        if not distance < 0.5:
            return "Unknown", distance
        stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
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
