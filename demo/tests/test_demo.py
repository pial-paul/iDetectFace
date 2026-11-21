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

    def start_app(self, language):
        app = AppTest.from_file(APP).run()
        self.assertFalse(app.get("camera_input"))
        if language == "বাংলা":
            app.segmented_control[0].set_value(language).run()
        app.button[0].click().run()
        app.radio[0].set_value("upload").run()
        return app

    def test_guided_check_in_in_both_languages(self):
        from demo.interface_text import TEXT

        for language, words in TEXT.items():
            with self.subTest(language=language):
                app = self.start_app(language)
                responses = [(np.zeros(128), digest) for digest in (b"a", b"b", b"b")]
                with patch("demo.session.encode_photo", side_effect=responses) as encode:
                    app.file_uploader[0].set_value(("one.jpg", b"a", "image/jpeg")).run()
                    self.assertEqual(app.success[0].value, words["ready"])
                    app.file_uploader[0].set_value(("two.jpg", b"b", "image/jpeg")).run()
                    self.assertEqual(app.success[0].value, words["Checked in"])
                    self.assertEqual(len(app.get("download_button")), 1)
                    app.run()
                    self.assertEqual(encode.call_count, 2)
                    next(b for b in app.button if b.label == words["again"]).click().run()
                    app.file_uploader[0].set_value(("two.jpg", b"b", "image/jpeg")).run()
                    self.assertEqual(app.success[0].value, words["Already checked in today"])
                    self.assertEqual(len(app.session_state["visit"].entries), 1)
                next(b for b in app.button if b.label == words["clear"]).click().run()
                self.assertFalse(app.exception)
                self.assertNotIn("visit", app.session_state)
                self.assertFalse(app.get("camera_input"))
                self.assertEqual(app.segmented_control[0].value, language)

    def test_retry_after_errors_and_language_switch(self):
        from demo.interface_text import TEXT
        from demo.session import PhotoError

        app = self.start_app("English")
        with patch(
            "demo.session.encode_photo",
            side_effect=[
                PhotoError("none"),
                (np.zeros(128), b"a"),
                (np.zeros(128), b"a"),
                (np.ones(128), b"b"),
            ],
        ) as encode:
            app.file_uploader[0].set_value(("empty.jpg", b"empty", "image/jpeg")).run()
            self.assertEqual(app.warning[0].value, TEXT["English"]["none"])
            app.segmented_control[0].set_value("বাংলা").run()
            self.assertEqual(app.warning[0].value, TEXT["বাংলা"]["none"])
            self.assertEqual(encode.call_count, 1)
            app.radio[0].set_value("upload").run()
            for _ in range(2):
                app.file_uploader[0].set_value(("one.jpg", b"a", "image/jpeg")).run()
            self.assertEqual(app.warning[0].value, TEXT["বাংলা"]["same"])
            app.file_uploader[0].set_value(("two.jpg", b"b", "image/jpeg")).run()
            self.assertEqual(app.warning[0].value, TEXT["বাংলা"]["Unknown"])
            self.assertFalse(app.session_state["visit"].entries)
            self.assertFalse(app.exception)
