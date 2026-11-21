"""OpenCV capture, enrollment and nearest-neighbor recognition."""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional

import cv2
import face_recognition as faces
import numpy as np
from filelock import FileLock
from PIL import Image

from .identity import enrollment_stem, identity_key

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def images(directory):
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


class Match(NamedTuple):
    box: tuple  # top, right, bottom, left in original frame pixels
    name: Optional[str]
    distance: float


class Gallery:
    def __init__(self, names, encodings):
        self.names = tuple(names)
        self.encodings = np.asarray(encodings, dtype=float)

    @classmethod
    def load(cls, directory: Path):
        names, encodings = [], []
        directory.mkdir(parents=True, exist_ok=True)
        for path in images(directory):
            name = path.stem.replace("_", " ")
            try:
                if not name.strip() or name.lstrip()[0] in "=+-@" or not name.isprintable():
                    raise ValueError("unsafe identity label")
                rgb = faces.load_image_file(str(path))
                locations = faces.face_locations(rgb)
                if len(locations) != 1:
                    raise ValueError(f"expected one face, found {len(locations)}")
                (vector,) = faces.face_encodings(rgb, locations)
            except (OSError, RuntimeError, ValueError, Image.DecompressionBombError) as error:
                print(f"Skipping {path.name}: {error}")
                continue
            names.append(name)
            encodings.append(vector)
        return cls(names, encodings)

    def nearest(self, vector, threshold):
        distances = faces.face_distance(self.encodings, vector)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        return (self.names[index] if distance < threshold else None), distance

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
        with NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            temp_path.write_bytes(jpeg.tobytes())
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
                for landmarks in faces.face_landmarks(rgb, boxes, model="small"):
                    for points in landmarks.values():
                        for point in points:
                            cv2.circle(preview, point, 1, (30, 100, 255), -1)
            prompt = "SPACE to capture" if len(boxes) == 1 else "Show exactly one face"
            label(preview, prompt + " | ESC to cancel", (10, 25))
            cv2.imshow("Add New Face", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
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
