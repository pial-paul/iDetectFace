# Portfolio descriptions

## Software engineering résumé

**iDetectFace | Python, OpenCV, NumPy, dlib, CSV, GitHub Actions**

- Built a local webcam attendance system that matches 128-dimensional face embeddings, rejects out-of-threshold observations and records each participant's first check-in per day.
- Reduced median repeated-check latency from 4.194 ms to 0.267 ms on a synthetic 10,000-row CSV by indexing daily attendance and coordinating writers with an interprocess file lock.
- Added 26 automated tests covering concurrent writes, midnight rollover, legacy CSVs, enrollment validation, recognition decisions and camera failures. Configured GitHub Actions for Windows and Linux checks.

## Research or graduate application

- Developed an open-set face identification prototype with a configurable Euclidean-distance threshold and a held-out evaluation tool that reports false acceptances, false rejections, wrong identities and detection failures by cohort.
- Designed a reproducible experiment comparing two attendance lookup strategies across 100, 1,000 and 10,000 synthetic records. Quantified where locking overhead outweighed indexing gains on the test machine.

## Single-line version

Built a Python/OpenCV face recognition attendance system with daily deduplication and 26 automated tests, reducing median repeat-check latency by 93.6% on a 10,000-row synthetic CSV benchmark.

## Demonstrate it in an interview

1. Open the architecture diagram and trace one camera frame through the threshold decision to the register. Explain why an embedding distance cannot be interpreted directly as a probability.
2. Run `python -m benchmarks.duplicate_lookup`. Discuss why the short CSV favors the scan and why the index needs rebuilding when the local date changes.
3. Show the concurrent-writer regression test and the single-face enrollment check. Explain the separate failure costs of assigning the wrong person and missing a check-in.
4. For a research interview, open the evaluation protocol and explain how enrollment, calibration and held-out sessions are separated. Quote recognition or field-impact metrics only after collecting those measurements.

The quantitative résumé claims come from the [recorded benchmark and evaluation protocol](evaluation.md). The test count comes from `python -m unittest discover -s tests -v`.
