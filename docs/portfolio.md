# Portfolio descriptions

## Software engineering résumé

**iDetectFace | Python, OpenCV, NumPy, dlib, CSV, GitHub Actions**

- Built a local webcam attendance system that matches 128-dimensional face embeddings, rejects out-of-threshold observations and records each participant's first check-in per day.
- Reduced median repeated-check latency from 4.449 ms to 0.018 ms on a synthetic 10,000-row CSV by caching the last confirmed sighting and validating the register version before taking a file lock.
- Added 36 automated tests covering concurrent writes, midnight rollover, legacy CSVs, enrollment validation, recognition decisions and camera failures. Configured GitHub Actions for Windows and Linux checks.

## Research or graduate application

- Developed an open-set face identification prototype with a configurable Euclidean-distance threshold and a held-out evaluation tool that reports false acceptances, false rejections, wrong identities and detection failures by cohort.
- Designed a reproducible experiment comparing two attendance lookup strategies across 100, 1,000 and 10,000 synthetic records. Quantified the cost of repeat sightings with an unchanged register.

## Single-line version

Built a Python/OpenCV face recognition attendance system with daily deduplication and 36 automated tests, reducing median repeat-check latency by 99.6% on a 10,000-row synthetic CSV benchmark.

## Demonstrate it in an interview

1. Open the architecture diagram and trace one camera frame through the threshold decision to the register. Explain why an embedding distance cannot be interpreted directly as a probability.
2. Run `python -m benchmarks.duplicate_lookup`. Discuss how the matching row's position affects a scan and why the index needs rebuilding when the local date changes.
3. Show the concurrent-writer regression test and the single-face enrollment check. Explain the separate failure costs of assigning the wrong person and missing a check-in.
4. For a research interview, open the evaluation protocol and explain how enrollment, calibration and held-out sessions are separated. Quote recognition or field-impact metrics only after collecting those measurements.

The quantitative résumé claims come from the [recorded benchmark and evaluation protocol](evaluation.md). The test count comes from `python -m unittest discover -s tests -v`.
