# AtomWorld

AtomWorld combines memory-based policies for BCC Fe–Cu, FCC Ni–Cr–Al and
HCP Zr–H with an independent Poisson/MFPT calibration layer.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Full policy training requires compatible PyTorch/PyG dependencies.
See [setup and training](docs/reproduction.md).

## Train

The Full model uses action history and recurrent memory. Training budgets and
material settings are configurable.

```bash
python -m atomworld.train_bcc --help
python -m atomworld.backends.fcc.train --help
python -m atomworld.backends.hcp.train_hcp --help
```

## Calibrate

The calibration layer is independent of policy optimization.

```bash
python -m atomworld.cli collect --help
python -m atomworld.cli fit --help
python -m atomworld.residence_clock --help
python -m atomworld.cli verify --samples 5000
```

Poisson/MFPT potential differences measure signed progress. Conditional
residence calibration measures local waiting times; physical dynamics under a
biased policy require separate validation. See [methods](docs/methods.md).

## License

MIT. Third-party attribution is provided in [NOTICE.md](NOTICE.md).
