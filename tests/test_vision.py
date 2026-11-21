import io
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from benchmarks.evaluate import evaluate
from idetectface import vision
from idetectface.attendance import AttendanceLog


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.vector = np.zeros(128)
        self.gallery = vision.Gallery(["demo"], [self.vector])

    def test_threshold_is_strict_and_distance_is_computed_once(self):
        probe = self.vector.copy()
        probe[0] = 0.5
        with patch.object(
            vision.faces, "face_distance", wraps=vision.faces.face_distance
        ) as distance:
            self.assertEqual(self.gallery.nearest(probe, 0.5), (None, 0.5))
            distance.assert_called_once()
        self.assertEqual(self.gallery.nearest(probe, 0.51), ("demo", 0.5))

    def test_nearest_identity_wins_regardless_of_order(self):
        gallery = vision.Gallery(["far", "near"], [np.ones(128), self.vector])
        self.assertEqual(gallery.nearest(self.vector, 0.5), ("near", 0.0))

    def test_native_detector_on_empty_image(self):
        self.assertEqual(self.gallery.recognize(self.frame, 0.5), [])

    def test_native_model_assets_generate_a_finite_descriptor(self):
        box = (15, 105, 95, 25)
        self.assertEqual(len(vision.faces.face_landmarks(self.frame, box)), 5)
        (descriptor,) = vision.faces.face_encodings(self.frame, [box])
        self.assertEqual(descriptor.shape, (128,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_rotated_color_channels_and_rescaled_boxes(self):
        self.frame[:, :, 0] = 250
        with (
            patch.object(vision.faces, "face_locations", return_value=[(2, 30, 10, 4)]) as detect,
            patch.object(vision.faces, "face_encodings", return_value=[self.vector]),
        ):
            (match,) = self.gallery.recognize(self.frame, 0.5)
        rgb = detect.call_args.args[0]
        self.assertEqual(rgb.shape, (60, 80, 3))
        self.assertEqual(rgb[0, 0].tolist(), [0, 0, 250])
        self.assertEqual(match, ((4, 60, 20, 8), "demo", 0.0))

    def test_gallery_skips_corrupt_empty_and_group_images(self):
        for name in ("a.jpg", "b.png", "c.jpeg", "d_demo.jpg"):
            (self.directory / name).touch()
        with (
            patch.object(
                vision, "read_image", side_effect=[OSError("broken")] + [(self.frame, b"id")] * 3
            ),
            patch.object(vision.faces, "face_locations", side_effect=[[], [1, 2], [1]]),
            patch.object(vision.faces, "face_encodings", return_value=[self.vector]),
            redirect_stdout(io.StringIO()),
        ):
            gallery = vision.Gallery.load(self.directory)
        self.assertEqual(gallery.names, ("d demo",))

    def test_enrollment_collision_across_formats_preserves_original(self):
        path = self.directory / "Demo One.PNG"
        path.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            vision.save_portrait(self.directory, "demo_one", self.frame)
        self.assertEqual(path.read_bytes(), b"existing")
        self.assertFalse((self.directory / "demo_one.jpg").exists())

    def test_failed_jpeg_does_not_leave_partial_image(self):
        with (
            patch.object(vision.cv2, "imencode", return_value=(False, None)),
            self.assertRaises(OSError),
        ):
            vision.save_portrait(self.directory, "demo", self.frame)
        self.assertEqual(vision.images(self.directory), [])

    def test_camera_open_failure_releases_device(self):
        cap = Mock()
        cap.isOpened.return_value = False
        with (
            patch.object(vision.cv2, "VideoCapture", return_value=cap),
            patch.object(vision.cv2, "destroyAllWindows"),
            self.assertRaisesRegex(RuntimeError, "camera 2"),
        ):
            with vision.camera(2):
                self.fail("Unexpected open camera")
        cap.release.assert_called_once()

    def fake_display(self, stack, keys):
        cap = Mock()
        cap.read.side_effect = lambda: (True, self.frame.copy())
        stack.enter_context(patch.object(vision.cv2, "VideoCapture", return_value=cap))
        for method in ("namedWindow", "imshow", "destroyAllWindows"):
            stack.enter_context(patch.object(vision.cv2, method))
        stack.enter_context(patch.object(vision.cv2, "getWindowProperty", return_value=1.0))
        stack.enter_context(patch.object(vision.cv2, "waitKey", side_effect=keys))
        return cap

    def test_group_capture_is_rejected_then_single_face_saved(self):
        box = (10, 70, 60, 20)
        with ExitStack() as stack:
            cap = self.fake_display(stack, [32, 32])
            stack.enter_context(
                patch.object(vision.faces, "face_locations", side_effect=[[box, box], [box], [box]])
            )
            stack.enter_context(patch.object(vision.faces, "face_landmarks", return_value=[]))
            vision.enroll(self.directory, 0, "Demo A")
        self.assertEqual(cap.read.call_count, 2)
        cap.release.assert_called_once()
        saved, _ = vision.read_image(self.directory / "demo_a.jpg")
        self.assertTrue(np.all(saved == 0), "preview annotations leaked into the saved portrait")

    def test_cadence_unknown_rejection_and_daily_log(self):
        box = (5, 40, 35, 10)
        known = vision.Match(box, "demo", 0.2)
        unknown = vision.Match(box, None, 0.7)
        log = AttendanceLog(self.directory / "attendance.csv")
        with ExitStack() as stack:
            cap = self.fake_display(stack, [0, 0, 0, 0, ord("q")])
            stack.enter_context(patch.object(vision.Gallery, "load", return_value=self.gallery))
            recognize = stack.enter_context(
                patch.object(self.gallery, "recognize", side_effect=[[unknown], [known], [known]])
            )
            labels = stack.enter_context(patch.object(vision, "label"))
            vision.run(self.directory, log, 0, 0.5, 2)
        self.assertEqual(recognize.call_count, 3)
        self.assertEqual([r.name for r in log.records()], ["demo"])
        captions = [call.args[1] for call in labels.call_args_list]
        self.assertIn("Unknown | d=0.700", captions)
        self.assertIn("demo | present | d=0.200", captions)
        cap.release.assert_called_once()

    def test_camera_disconnect_releases_device(self):
        with ExitStack() as stack:
            cap = self.fake_display(stack, [])
            cap.read.side_effect = [(False, None)] * 3
            stack.enter_context(patch.object(vision.Gallery, "load", return_value=self.gallery))
            with self.assertRaisesRegex(RuntimeError, "stopped delivering"):
                vision.run(self.directory, None, 0, 0.5, 2)
        cap.release.assert_called_once()

    def test_evaluation_counts_all_probe_outcomes(self):
        gallery_dir = self.directory / "gallery"
        gallery_dir.mkdir()
        manifest = self.directory / "probes.csv"
        lines = ["image,expected"]
        outcomes = [[vision.Match((), name, 0.3)] for name in ("demo", "other", None, "demo", None)]
        outcomes.extend([[], [vision.Match((), None, 0.8)] * 2])
        expected = ["demo", "demo", "demo", "@unknown", "@unknown", "demo", "@unknown"]
        for i, target in enumerate(expected):
            vision.cv2.imwrite(str(self.directory / f"p{i}.png"), self.frame + i)
            lines.append(f"p{i}.png,{target}")
        manifest.write_text("\n".join(lines), encoding="utf-8")
        with (
            patch.object(vision.Gallery, "load", return_value=self.gallery),
            patch.object(self.gallery, "recognize", side_effect=outcomes),
        ):
            result = evaluate(gallery_dir, manifest, 0.5)
        self.assertEqual(
            result,
            {
                "known": 4,
                "unknown": 3,
                "correct_id": 1,
                "wrong_id": 1,
                "false_reject": 1,
                "false_accept": 1,
                "correct_reject": 1,
                "known_detection_failure": 1,
                "unknown_detection_failure": 1,
            },
        )

    def test_evaluation_rejects_renamed_copy_of_enrollment_image(self):
        gallery_dir = self.directory / "gallery"
        gallery_dir.mkdir()
        vision.cv2.imwrite(str(gallery_dir / "demo.png"), self.frame)
        (self.directory / "probe.png").write_bytes((gallery_dir / "demo.png").read_bytes())
        manifest = self.directory / "probes.csv"
        manifest.write_text("image,expected\nprobe.png,demo\n", encoding="utf-8")
        _, digest = vision.read_image(gallery_dir / "demo.png")
        gallery = vision.Gallery(["demo"], [self.vector], [digest])
        with (
            patch.object(vision.Gallery, "load", return_value=gallery),
            self.assertRaisesRegex(ValueError, "reuses an enrollment"),
        ):
            evaluate(gallery_dir, manifest, 0.5)

    def test_tied_identities_are_unknown(self):
        ambiguous = vision.Gallery(["one", "two"], [self.vector, self.vector])
        self.assertEqual(ambiguous.nearest(self.vector, 0.5), (None, 0.0))
        same_id = vision.Gallery(["one", "ONE"], [self.vector, self.vector])
        self.assertEqual(same_id.nearest(self.vector, 0.5), ("one", 0.0))
        self.assertIsNone(vision.Gallery([], []).nearest(self.vector, 0.5)[0])

    def test_unusable_compressed_portrait_is_not_saved(self):
        with self.assertRaisesRegex(ValueError, "saved JPEG"):
            vision.save_portrait(self.directory, "demo", self.frame)
        self.assertEqual(vision.images(self.directory), [])

    def test_closing_preview_releases_camera_without_enrolling(self):
        with ExitStack() as stack:
            cap = self.fake_display(stack, [0])
            stack.enter_context(patch.object(vision.cv2, "getWindowProperty", return_value=0))
            vision.enroll(self.directory, 0, "demo")
        cap.release.assert_called_once()
        self.assertEqual(vision.images(self.directory), [])

    def test_decoding_enforces_pixel_and_file_limits(self):
        from idetectface import images

        path = self.directory / "image.png"
        vision.cv2.imwrite(str(path), self.frame)
        with patch.object(images, "MAX_PIXELS", 10), self.assertRaises(ValueError):
            images.read_image(path)
        with patch.object(images, "MAX_BYTES", 10), self.assertRaises(ValueError):
            images.read_image(path)

    def test_manifest_stays_within_its_dataset(self):
        folder = self.directory / "dataset"
        folder.mkdir()
        manifest = folder / "probes.csv"
        for name in ("../another-dataset.png", str(self.directory / "outside.png")):
            manifest.write_text(f"image,expected\n{name},@unknown\n", encoding="utf-8")
            with (
                self.subTest(name=name),
                patch.object(vision.Gallery, "load", return_value=self.gallery),
                self.assertRaisesRegex(ValueError, "manifest folder"),
            ):
                evaluate(folder, manifest, 0.5)
