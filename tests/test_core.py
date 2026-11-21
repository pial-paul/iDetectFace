import io
import multiprocessing
import sys
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from idetectface.attendance import AttendanceLog
from idetectface.identity import enrollment_stem
from opencv_attendance import main

DAY = datetime(2025, 3, 9, 23, 59, 59)


def write_in_worker(path):
    return AttendanceLog(path).mark("demo 01", DAY)


class RegisterTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "attendance.csv"
        self.log = AttendanceLog(self.path)

    def test_local_midnight_and_restart(self):
        self.assertTrue(self.log.mark("Demo 01", DAY))
        self.assertFalse(AttendanceLog(self.path).mark("demo 01", DAY))
        self.assertTrue(self.log.mark("demo 01", DAY + timedelta(seconds=1)))
        self.assertEqual(self.log.records(DAY.date()), [("Demo 01", "2025-03-09", "23:59:59")])

    def test_utc_observations_and_today_report_follow_dhaka_midnight(self):
        before = datetime.fromisoformat("2025-12-31T17:59:59+00:00")
        after = before + timedelta(seconds=1)
        self.assertTrue(self.log.mark("Demo", before))
        self.assertTrue(self.log.mark("Demo", after))
        self.assertFalse(self.log.mark("Demo", after + timedelta(hours=6)))
        from idetectface.clock import bangladesh_time

        output = io.StringIO()
        with (
            patch("idetectface.cli.bangladesh_time", return_value=bangladesh_time(after)),
            redirect_stdout(output),
        ):
            self.assertEqual(main(["--data-dir", str(self.path.parent), "records"]), 0)
        self.assertIn("2026-01-01  |  1 present", output.getvalue())
        self.assertIn("00:00:00  Demo", output.getvalue())
        self.assertNotIn("2025-12-31", output.getvalue())

    def test_cached_index_avoids_reopening_csv_for_duplicates(self):
        self.log.mark("Demo", DAY)
        with patch.object(self.log, "_rows", side_effect=AssertionError("unexpected scan")):
            for _ in range(5):
                self.assertFalse(self.log.mark("Demo", DAY))

    def test_another_writer_invalidates_cached_index(self):
        self.log.mark("Demo A", DAY)
        AttendanceLog(self.path).mark("Demo B", DAY)
        self.assertFalse(self.log.mark("Demo B", DAY))

    def test_same_identity_is_recorded_again_after_register_replacement(self):
        self.log.mark("Demo A", DAY)
        fresh = self.path.with_suffix(".new")
        fresh.write_bytes(b"Name,Date,Time\r\n")
        fresh.replace(self.path)
        self.assertTrue(self.log.mark("Demo A", DAY))
        self.assertEqual(len(self.log.records()), 1)

    def test_processes_share_one_daily_mark(self):
        with ProcessPoolExecutor(3, mp_context=multiprocessing.get_context("spawn")) as pool:
            self.assertEqual(sum(pool.map(write_in_worker, [self.path] * 12)), 1)
        self.assertEqual(len(self.log.records()), 1)

    def test_legacy_header_bom_csv_quoting_and_missing_final_newline(self):
        self.path.write_text(
            'Name,Date,Time\n"Demo, One",2025-03-09,08:30:00', encoding="utf-8-sig"
        )
        self.assertFalse(self.log.mark("DEMO, ONE", DAY))
        self.assertTrue(self.log.mark("Zoë", DAY))
        self.assertEqual([r.name for r in self.log.records()], ["Demo, One", "Zoë"])

    def test_malformed_register_is_preserved(self):
        for data in (
            "",
            "Name,Time\n",
            "Name,Date,Time\nDemo,2025-03-09\n",
            "Name,Date,Time\nDemo,2025-02-30,09:00:00\n",
        ):
            with self.subTest(data=data):
                self.path.write_text(data, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.log.mark("Demo", DAY)
                self.assertEqual(self.path.read_text(encoding="utf-8"), data)

    def test_failure_does_not_poison_index(self):
        with patch.object(self.log, "_append", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.log.mark("Demo", DAY)
        self.assertTrue(self.log.mark("Demo", DAY))

    def test_sync_failure_rolls_back_both_first_and_existing_registers(self):
        with (
            patch("idetectface.attendance.os.fsync", side_effect=OSError("write failed")),
            self.assertRaises(OSError),
        ):
            self.log.mark("A", DAY)
        self.assertFalse(self.path.exists())
        self.log.mark("A", DAY)
        before = self.path.read_bytes()
        with (
            patch("idetectface.attendance.os.fsync", side_effect=OSError("write failed")),
            self.assertRaises(OSError),
        ):
            self.log.mark("B", DAY)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.log.mark("B", DAY))

    def test_register_rejects_unsafe_historical_label(self):
        raw = "Name,Date,Time\n=demo,2025-03-09,09:30:00\n"
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.log.records()
        self.assertEqual(self.path.read_text(encoding="utf-8"), raw)

    def test_unicode_filename_fits_filesystem_byte_limit(self):
        with self.assertRaises(ValueError):
            enrollment_stem("𠀀" * 80)

    def test_missing_report_is_empty(self):
        self.assertEqual(self.log.records(date.today()), [])
        self.assertFalse(self.path.exists())

    def test_unsafe_names_cannot_create_csv(self):
        for name in ("", "  ", "=2+2", " @demo", "Demo\n01"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.log.mark(name, DAY)
        self.assertFalse(self.path.exists())

    def test_stem_is_portable_and_keeps_unicode(self):
        self.assertEqual(enrollment_stem("  Zoë  Demo  "), "zoë_demo")
        self.assertEqual(enrollment_stem("Demo_01"), "demo_01")
        for name in ("../demo", r"C:\demo", "con", "LPT1", "a/b", "=f", "01\n02"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                enrollment_stem(name)

    def test_reports_work_without_camera_libraries(self):
        self.log.mark("Demo", DAY)
        with patch.dict(sys.modules, {"cv2": None, "face_recognition": None}):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--data-dir", str(self.path.parent), "records", "--all"]), 0)
            self.assertIn("2025-03-09  |  1 present\n  23:59:59  Demo", output.getvalue())

    def test_menu_preserves_by_date_navigation(self):
        self.log.mark("Demo", DAY)
        output = io.StringIO()
        with (
            patch("builtins.input", side_effect=["3", "2", "2025-03-09", "4"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(["--data-dir", str(self.path.parent)]), 0)
        self.assertIn("23:59:59  Demo", output.getvalue())

    def test_invalid_cli_options_fail_before_capture(self):
        for args in (
            ["--every", "0", "run"],
            ["--threshold", "nan", "run"],
            ["records", "--date", "2025-02-30"],
        ):
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as err,
            ):
                main(args)
            self.assertEqual(err.exception.code, 2)
