# Development and release checks

Use the CPython 3.11 environment in the [setup guide](README.md). Application code lives in `idetectface/`, synthetic regression tests in `tests/`, and experiments in `benchmarks/`. The top-level script preserves the original command line.

## Test a change

```sh
python -m pip install ruff==0.11.13
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
python -m benchmarks.duplicate_lookup
```

The automated tests require the native libraries and packaged models. They run with temporary data and controlled camera frames. For a device or recognition change, also follow the [physical-camera checks](docs/evaluation.md).

## Change dependencies

Edit the project metadata first. Regenerate the runtime lock with a Python 3.11 environment and uv, preserving hashes for the supported platforms.

```sh
uv pip compile pyproject.toml --python-version 3.11 --universal --generate-hashes --no-header --no-annotate --upgrade -o requirements.txt
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-deps .
python -m pip check
python -I -m idetectface --help
```

Run the advisory check in a separate tools environment. Also check the underlying dlib version when updating the community wheel.

```sh
uvx --from pip-audit==2.10.1 pip-audit --no-deps --disable-pip -r requirements.txt
```

## Review a release

1. Run the complete suite on Windows and Linux. Inspect the wheel from outside the source tree and exercise the installed console command.
2. Review the actual Git tree and wheel contents. Keep portraits, recordings, credentials, manifests and attendance exports out of both.
3. Run `gitleaks git --redact --log-opts="--all" .` and scan a clean export with `gitleaks dir --redact`. A secret scanner complements the manual data review.
4. Publish a fast-forward commit and verify its remote SHA and Actions results. Record device and dataset limitations separately from software test outcomes.
