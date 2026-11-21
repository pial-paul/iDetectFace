import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from demo.session import Visit, encode_photo

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


class DemoTests(unittest.TestCase):
    def test_threshold_duplicate_utc_rollover_and_isolation(self):
        vector = np.zeros(128)
        visit = Visit(vector, b"enrolled")
        other = Visit(vector.copy(), b"other")
        probe = vector.copy()
        probe[0] = 0.5
        self.assertEqual(visit.check_in(probe, b"test")[0], "Unknown")
        self.assertFalse(visit.entries)
        with self.assertRaises(ValueError):
            visit.check_in(vector, b"enrolled")
        for day, expected in [
            (1, "Checked in"),
            (1, "Already checked in today"),
            (2, "Checked in"),
        ]:
            now = datetime(2026, 1, day, tzinfo=timezone.utc)
            self.assertEqual(visit.check_in(vector, b"test", now)[0], expected)
        self.assertEqual(len(visit.csv_text().splitlines()), 3)
        self.assertFalse(other.entries)

    def test_invalid_uploads_fail_without_creating_state(self):
        for data in (b"", b"a" * (4 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                encode_photo(data)

    def test_app_enroll_check_in_export_and_reset(self):
        app = AppTest.from_file(APP).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.get("camera_input"))
        app.checkbox[0].check().run()
        app.radio[0].set_value("Upload").run()
        with patch(
            "demo.session.encode_photo", side_effect=[(np.zeros(128), b"a"), (np.zeros(128), b"b")]
        ):
            app.file_uploader[0].set_value(("one.jpg", b"a", "image/jpeg")).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            app.radio[0].set_value("Upload").run()
            app.file_uploader[0].set_value(("two.jpg", b"b", "image/jpeg")).run()
            app.button[0].click().run()
        self.assertIn("Checked in", app.success[0].value)
        self.assertEqual(len(app.get("download_button")), 1)
        next(b for b in app.button if b.label == "Clear my session").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("visit", app.session_state)
        self.assertFalse(app.checkbox[0].value)
