# Evaluation and reproduction

## Repeated attendance checks

From the project root, run `python -m benchmarks.duplicate_lookup`. The script creates temporary synthetic CSVs, puts one identity for the selected day in the final row, and checks that identity repeatedly. Both implementations must return “already present” on the warm-up. Each reported value is the median of seven batch durations divided by 50 checks per batch.

| CSV data rows | Original scan, µs/check | Cached repeat, µs/check |
| ---: | ---: | ---: |
| 100 | 85.6 | 21.4 |
| 1,000 | 520.8 | 21.2 |
| 10,000 | 4,448.7 | 17.7 |

Measured with 64-bit CPython 3.11.13, Windows, an Intel Core i5-1235U and filelock 4.2.12. The original column reproduces the baseline's serial scan and early return. The second calls `AttendanceLog.mark` with the same last-confirmed identity. Both receive a fixed observation day. Setup, first writes, disk sync and index construction finish before timing starts. [Benchmark source](../benchmarks/duplicate_lookup.py)

A repeat checks file metadata before returning. Alternating identities acquire the lock and consult the daily index. A concurrent write or date change triggers an index refresh. An early-row match would reduce the cost of the original scan. Disk cache, hardware and competing processes affect these figures.

## Recognition evaluation

The repository includes a runner and a labeling protocol. An accuracy estimate requires separately collected, consented probe images. The published benchmark above measures storage behavior.

1. Enroll consenting participants under unique IDs. Collect one-person probe photos in separate sessions with different lighting, pose and distance. Include people who opted into evaluation and have no gallery entry. Keep calibration and final test sessions separate. Count participants as well as photos, since many photos of one person provide limited independent evidence.
2. Keep private images and manifests under `.local-data/` or outside the checkout. Prepare a UTF-8 CSV headed `image,expected`. Keep probes inside the manifest directory tree. Supply relative paths to regular JPEG or PNG files up to 16 MiB and 16 megapixels each. A known label must match the loaded gallery name using spaces for filename underscores. Use `@unknown` explicitly for a person outside the gallery. Example labels and paths below are fictional.

   ```csv
   image,expected
   probes/session-b-01.jpg,demo-01
   probes/session-c-02.jpg,@unknown
   ```

3. Run the same half-resolution detector, encoder and threshold decision used on webcam frames. Repeat on calibration data at candidate thresholds, for example 0.45, 0.50 and 0.55. Choose a threshold according to the cost of marking the wrong participant and the manual-review capacity. Freeze it before opening the final test manifest.

   ```sh
   python -m benchmarks.evaluate --gallery .local-data/known_faces --probes .local-data/calibration.csv --threshold 0.5
   python -m benchmarks.evaluate --gallery .local-data/known_faces --probes .local-data/test.csv --threshold 0.5
   ```

4. Report each outcome count and its denominator. For **known probes**, the outcomes are correct identity, wrong identity, false rejection, or detection failure. For **unknown probes**, they are correct rejection, false acceptance, or detection failure. Zero or multiple detections count as a detection failure in this single-person protocol. Each percentage uses all probes in its cohort, including detection failures. Empty cohorts have no rate.
5. Publish the capture conditions, gallery size, participant count, test counts and chosen threshold alongside results. The runner rejects byte-identical copies of enrollment images and duplicate probes within a manifest. The evaluator must also exclude recompressed copies, adjacent video frames and cross-split leakage. Review errors by lighting and distance before attributing them to the matcher.

## Physical-camera acceptance check

Use a fresh private register and an organizer observing the session.

| Action | Expected observation |
| --- | --- |
| Attempt enrollment with zero or two faces, then press Space. | The prompt keeps requesting one face. No portrait is saved. |
| Enroll one participant and start attendance. Have them exit and re-enter the frame. | An accepted match shows its distance. The CSV contains one row for that participant that day. |
| Restart capture with the same data folder, then repeat. | A recognized participant shows as present. The original timestamp stays intact. |
| Have an unenrolled volunteer appear. | Check for an Unknown label and inspect the register for a false acceptance. Record a detection failure separately if there is no box. |
| Press q, return to the menu and open the camera again. | The prior session releases the device. |

For a frame-rate claim, record the camera, resolution, processor, gallery size, number of simultaneous faces and `--every` setting. Time capture-to-display over a defined interval after gallery loading. Report frames displayed per second separately from frames analyzed per second and include tail latency. Neither number has been measured here.
