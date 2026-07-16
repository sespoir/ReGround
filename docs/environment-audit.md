# Environment audit

The source machine dependency stack was inspected on 2026-07-16. New project
installations use a Conda environment named `reground`.

| Component | Observed version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.9.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| Transformers | 4.57.6 |
| vLLM | 0.11.2 |
| OpenAI SDK | 2.17.0 |
| pandas | 2.3.3 |
| Pillow | 11.3.0 |
| requests | 2.32.5 |

`conda env export --from-history` contains only Python 3.10, `gxx_linux-64`
11.4, and `sysroot_linux-64` 2.28 because most runtime packages were installed
with pip. `environment.yml` therefore defines the portable base environment;
`bootstrap.sh` installs VLMEvalKit's client dependencies, and
`requirements-server.txt` records the validated serving stack.

The original Conda installation prints a `conda-libmamba-solver` / `libarchive`
loading warning. It did not prevent inspection, but the Conda installation
should be repaired before using it to create a fresh environment.

The repository standardizes on `reground`. Set `REGROUND_CONDA_ENV` only when
reusing an existing environment whose package versions and custom VLMEvalKit
adapter have been verified.
