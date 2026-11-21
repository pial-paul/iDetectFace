"""Identity rules shared by enrollment, matching and the daily register."""

import unicodedata
from pathlib import PureWindowsPath


def identity_key(name: str) -> str:
    return unicodedata.normalize("NFC", " ".join(name.split()).casefold())


def enrollment_stem(name: str) -> str:
    name = unicodedata.normalize("NFC", name.strip())
    if not name or len(name) > 80 or not name[0].isalnum():
        raise ValueError("Start with a letter or digit and use at most 80 characters.")
    for ch in name:
        if not (ch.isalnum() or unicodedata.category(ch).startswith("M") or ch in " '-_"):
            raise ValueError("Use letters, numbers, spaces, hyphens, underscores or apostrophes.")
    stem = "_".join(name.lower().replace("_", " ").split())
    if PureWindowsPath(stem + ".jpg").is_reserved():
        raise ValueError("Choose a name that is safe as a filename on Windows.")
    return stem
