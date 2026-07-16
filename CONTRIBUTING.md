# Contributing

Contributions that improve reproducibility, compatibility, tests, or
documentation are welcome.

## Development setup

Create the client environment from the repository root:

```bash
bash scripts/bootstrap.sh
conda activate reground
```

The repository adapter is installed into a pinned VLMEvalKit checkout. If the
adapter source changes, reinstall it before an integration test:

```bash
bash scripts/install_vlmevalkit_adapter.sh third_party/VLMEvalKit
```

## Required checks

Run the following checks before opening a pull request:

```bash
bash -n scripts/*.sh jobs/*.sbatch
ruff check src scripts tests
python -m py_compile src/*.py scripts/*.py tests/*.py
python tests/test_reground_payload.py
bash scripts/secret_scan.sh
git diff --check
```

Changes to the two-round protocol should include an offline contract test.
Changes to serving behavior should report the vLLM and CUDA versions used for
validation. Do not commit model weights, datasets, predictions, token logs,
cluster logs, credentials, or private endpoint details.

## Reporting issues

Please include a minimal reproducer, the VLMEvalKit commit, Python and vLLM
versions, relevant configuration values with secrets removed, and the exact
error message. Do not attach private images or unredacted model logs.
