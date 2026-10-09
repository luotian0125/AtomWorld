# Setup and training

## Environment

The tested BCC environment uses Python 3.10, PyTorch 2.6.0+cu124, PyG 2.6.1
and torch-cluster 1.6.3. Direct dependencies are pinned in
`requirements/reproduction.txt`.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install -r requirements/reproduction.txt
python -m pip install -e ".[training,dev]"
```

Install matching CPU or CUDA builds for your machine.

## Full BCC model

Set `EVENTS_PER_ROUND`, `ROUNDS` and `CU_ATOMS` for the desired training run.

```bash
export ATOMWORLD_RESULTS="$PWD/runs/models"
python -m atomworld.train_bcc --episode "$EVENTS_PER_ROUND" \
  --rounds "$ROUNDS" --cu "$CU_ATOMS" --size 50 --tag atomworld --seed 101
```

The default physical setting is 50³ unit cells, 600 K and four vacancies.
The model uses action history, recurrent memory, hidden size 256 and two GNN
layers. Effective settings are saved with the final checkpoint. Policy training
uses no event-time loss; residence calibration is fitted independently.

## Slurm

From a CUDA-enabled environment:

```bash
export ATOMWORLD_PYTHON="$(command -v python)"
python tools/submit_bcc.py --root runs/bcc --partition YOUR_GPU_PARTITION \
  --episode "$EVENTS_PER_ROUND" --rounds "$ROUNDS" --cu "$CU_ATOMS"
```

This submits one Full training job requesting one GPU, four CPU cores and at
most 24 hours. Outputs are written beneath the run root. Run GPU jobs through
your scheduler where required.

## Calibration and evaluation

```bash
python -m atomworld.residence_clock --root runs/bcc --cu "$CU_ATOMS" \
  --episode "$EVENTS_PER_ROUND" --rounds "$ROUNDS" --size 50
python -m atomworld.evaluate_bcc --help
```

Evaluation takes an explicit policy checkpoint, residence checkpoint and event
budget. Residence checkpoints are restricted to their physical domain.

## FCC and HCP

```bash
python -m atomworld.backends.fcc.train --help
python -m atomworld.backends.hcp.train_hcp --help
```

Material configurations are packaged under `atomworld/backends/*/configs`.
For MFPT collection, specify the absorbing target and supply independent
reference data through `atomworld.cli collect` and `atomworld.cli fit`.
