"""OpenCV capture, enrollment and nearest-neighbor recognition."""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional

import cv2
import numpy as np
from filelock import FileLock

from . import recognition as faces
from .identity import enrollment_stem, identity_key, validate_label
from .images import IMAGE_SUFFIXES, read_image


def images(directory):
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


class Match(NamedTuple):
    box: tuple  # top, right, bottom, left in original frame pixels
    name: Optional[str]
    distance: float


class Gallery:
    def __init__(self, names, encodings, fingerprints=()):
        self.names = tuple(names)
        self.encodings = np.asarray(encodings, dtype=float)
        self.fingerprints = frozenset(fingerprints)

    @classmethod
    def load(cls, directory: Path):
        names, encodings, fingerprints = [], [], set()
        directory.mkdir(parents=True, exist_ok=True)
        for path in images(directory):
            name = path.stem.replace("_", " ")
            try:
                validate_label(name)
                rgb, digest = read_image(path)
                locations = faces.face_locations(rgb)
                if len(locations) != 1:
                    raise ValueError(f"expected one face, found {len(locations)}")
                (vector,) = faces.face_encodings(rgb, locations)
            except (OSError, ValueError) as error:
                print(f"Skipping {path.name}: {error}")
                continue
            names.append(name)
            encodings.append(vector)
            fingerprints.add(digest)
        return cls(names, encodings, fingerprints)

    def nearest(self, vector, threshold):
        if not self.names:
            return None, float("inf")
        distances = faces.face_distance(self.encodings, vector)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if not distance < threshold:
            return None, distance
        tied = np.flatnonzero(distances == distance)
        if len(tied) > 1 and len({identity_key(self.names[i]) for i in tied}) > 1:
            return None, distance
        return self.names[index], distance

    def recognize(self, frame, threshold):
        small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = faces.face_locations(rgb)
        vectors = faces.face_encodings(rgb, locations)
        sy = frame.shape[0] / small.shape[0]
        sx = frame.shape[1] / small.shape[1]
        results = []
        for (top, right, bottom, left), vector in zip(locations, vectors):
            box = (round(top * sy), round(right * sx), round(bottom * sy), round(left * sx))
            results.append(Match(box, *self.nearest(vector, threshold)))
        return results


@contextmanager
def camera(index):
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}. Check permissions or try --camera 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        yield cap
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Headless OpenCV builds have no window manager.


def read_frame(cap):
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    raise RuntimeError("Camera stopped delivering frames. Reconnect it and retry.")


def label(frame, text, point, color=(255, 255, 255)):
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)


def window_closed(title):
    # Wayland does not expose HighGUI window properties. The quit keys still work.
    if cv2.currentUIFramework() == "WAYLAND":
        return False
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error as error:
        if error.code == cv2.Error.StsNullPtr:  # Win32 removes a closed window immediately.
            return True
        raise


def save_portrait(directory: Path, name: str, frame):
    stem = enrollment_stem(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (stem + ".jpg")
    with FileLock(str(directory / ".enrollment.lock"), timeout=5):
        key = identity_key(stem.replace("_", " "))
        if any(identity_key(p.stem.replace("_", " ")) == key for p in images(directory)):
            raise ValueError("This identity already has a portrait. Choose a unique name or ID.")
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            raise OSError("OpenCV could not encode the portrait.")
        stored = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if (
            stored is None
            or len(faces.face_locations(cv2.cvtColor(stored, cv2.COLOR_BGR2RGB))) != 1
        ):
            raise ValueError("The saved JPEG must contain one detectable face. Please retry.")
        temp = NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False)
        temp_path = Path(temp.name)
        try:
            with temp:
                temp.write(jpeg.tobytes())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
    return path


def enroll(directory, index, name):
    enrollment_stem(name)  # Reject invalid input before opening the device.
    with camera(index) as cap:
        while True:
            frame = read_frame(cap)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = faces.face_locations(rgb)
            preview = frame.copy()
            for top, right, bottom, left in boxes:
                cv2.rectangle(preview, (left, top), (right, bottom), (70, 190, 110), 2)
            if len(boxes) == 1:
                for point in faces.face_landmarks(rgb, boxes[0]):
                    cv2.circle(preview, point, 1, (30, 100, 255), -1)
            prompt = "SPACE to capture" if len(boxes) == 1 else "Show exactly one face"
            label(preview, prompt + " | ESC to cancel", (10, 25))
            cv2.imshow("Add New Face", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_closed("Add New Face"):
                return
            if key == 32 and len(boxes) == 1:
                path = save_portrait(directory, name, frame)
                print(f"Enrolled {path.stem.replace('_', ' ')}.")
                return


def run(directory, log, index, threshold, every):
    gallery = Gallery.load(directory)
    if not gallery.names:
        raise ValueError("No usable enrollment images. Enroll one participant first.")
    print(f"Loaded {len(gallery.names)} enrollment images. Press q to finish.")
    with camera(index) as cap:
        cv2.namedWindow("Face Recognition Attendance")
        frame_number = 0
        overlays = []
        while True:
            frame = read_frame(cap)
            now = datetime.now()
            if frame_number % every == 0:
                overlays = []
                for match in gallery.recognize(frame, threshold):
                    caption = "Unknown"
                    if match.name is not None:
                        marked = log.mark(match.name, now)
                        caption = f"{match.name} | {'recorded' if marked else 'present'}"
                        if marked:
                            print(f"{now:%Y-%m-%d %H:%M:%S}  {match.name}  recorded")
                    overlays.append((match, f"{caption} | d={match.distance:.3f}"))
            frame_number += 1
            for match, caption in overlays:
                top, right, bottom, left = match.box
                color = (70, 190, 110) if match.name is not None else (25, 160, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label(frame, caption, (left, max(18, top - 5)), color)
            label(frame, f"{now:%Y-%m-%d %H:%M:%S} | q to finish", (10, frame.shape[0] - 12))
            cv2.imshow("Face Recognition Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
            if window_closed("Face Recognition Attendance"):
                return
