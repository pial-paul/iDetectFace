"""CPU inference with the original five-point alignment and 128-D dlib model."""

from functools import cache
from importlib.metadata import distribution

import dlib
import numpy as np


def model_file(name):
    assets = distribution("face-recognition-models")
    path = assets.locate_file(f"face_recognition_models/models/{name}")
    if not path.is_file():
        raise RuntimeError(f"Missing {name}. Reinstall the runtime requirements.")
    return str(path)


@cache
def detector():
    return dlib.get_frontal_face_detector()


@cache
def predictor():
    return dlib.shape_predictor(model_file("shape_predictor_5_face_landmarks.dat"))


@cache
def encoder():
    return dlib.face_recognition_model_v1(model_file("dlib_face_recognition_resnet_model_v1.dat"))


def face_locations(rgb):
    height, width = rgb.shape[:2]
    return [
        (max(0, box.top()), min(width, box.right()), min(height, box.bottom()), max(0, box.left()))
        for box in detector()(rgb, 1)
    ]


def shape(rgb, box):
    top, right, bottom, left = box
    return predictor()(rgb, dlib.rectangle(left, top, right, bottom))


def face_landmarks(rgb, box):
    return [(point.x, point.y) for point in shape(rgb, box).parts()]


def face_encodings(rgb, boxes):
    return [np.asarray(encoder().compute_face_descriptor(rgb, shape(rgb, box), 1)) for box in boxes]


def face_distance(encodings, vector):
    return np.linalg.norm(encodings - vector, axis=1)
