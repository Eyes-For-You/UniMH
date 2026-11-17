# UniMH

This supplementary package contains the flow-matching teacher model, its
training loop, and evaluation code for ETH-UCY, SDD, and GigaTraj. It includes
MoE feature transformations and the auxiliary PTR/VQ-VAE training branch.
Teacher inference uses iterative flow sampling and does not execute PTR.

Datasets, model weights, student models, IMLE training, experiment queues,
paper sources, and plotting scripts are not included.

## Installation

Use Python 3.11. The dependencies match the environment used to check this
package; NumPy is pinned below version 2 for compatibility with PyTorch 2.0.1.
Install a PyTorch build appropriate for your CUDA environment if necessary.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python selfcheck.py
```

The self-check runs a small CPU forward/backward pass, an optimizer step, and
teacher sampling for each dataset configuration. It creates synthetic tensors
in memory only. These checks do not measure forecasting accuracy.
W&B is disabled in all supplied configurations and is not required.

## Contents

| Path | Purpose |
|---|---|
| `models/backbone_eth_ucy.py` | Shared trajectory teacher Transformer |
| `models/mds.py` | Sparse MoE transformations and routing |
| `models/ptr.py` | Prototype-guided auxiliary trajectory regularization |
| `models/flow_matching.py` | Flow-matching objective and iterative sampler |
| `models/context_encoder/`, `models/motion_decoder/` | Context encoder, time embedding, and motion decoder |
| `trainer/denoising_model_trainers.py` | Teacher optimization, EMA, checkpointing, and original evaluation |
| `data/dataloader_*.py` | Teacher data loading, normalization, rotation, and masks |
| `fm_eth.py`, `fm_sdd.py`, `fm_giga.py` | Dataset-specific teacher training entry points |
| `eval_eth.py`, `eval_sdd.py`, `eval_giga.py` | Original teacher evaluation entry points |
| `evaluate_teacher.py` | Fixed-checkpoint evaluation across input/output lengths |
| `cfg/` | Anonymous teacher configurations |
| `selfcheck.py` | Dataset-free software check |
| `SOURCE_MANIFEST.json` | Source hashes and packaging changes |
| `VALIDATION.json` | Packaging validation summary |

## Data preparation

Obtain the datasets separately and prepare the pickle files expected by the
loaders. No download, credentials, or data files are embedded in this package.
The default layout is:

```text
data/
  dataloader_eth_ucy.py
  dataloader_sdd.py
  dataloader_giga.py
  eth_ucy/original/
    eth/eth_train.pkl
    eth/eth_test.pkl
    hotel/hotel_train.pkl
    hotel/hotel_test.pkl
    univ/univ_train.pkl
    univ/univ_test.pkl
    zara1/zara1_train.pkl
    zara1/zara1_test.pkl
    zara2/zara2_train.pkl
    zara2/zara2_test.pkl
  sdd/original/
    sdd_train.pkl
    sdd_test.pkl
  giga/TUTR/giga/
    giga_train.pkl
    giga_test.pkl
```

The ETH-UCY and GigaTraj pickle files contain a dictionary with a `traj` array
of shape `[number_of_trajectories, total_frames, 2]`. ETH-UCY uses 8 observed
and 12 future frames; GigaTraj uses 8 observed and 90 future frames. SDD uses
a sequence of `(observed_trajectory, future_trajectory)` pairs, with 8 and 12
frames respectively. The loaders define the precise preprocessing contract.
Use only trusted pickle files.

All entry points accept an external data directory. For example,
`--data_dir /path/to/giga` in `fm_giga.py` corresponds to the directory
containing `TUTR/`; `--data-dir` is the equivalent option in
`evaluate_teacher.py`. Normalization is recovered from the training split
before loading the test split.

## Teacher training

Run commands from this package's root directory. The training flags below
make the rotation, input scaling, and tied-noise settings explicit.

```bash
python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset eth \
  --data_dir data/eth_ucy --data_source original \
  --rotate --rotate_time_frame 6 --tied_noise --fm_in_scaling \
  --use_vqvae --fix_random_seed --seed 42 --checkpt_freq 1 --exp teacher

python fm_sdd.py --cfg cfg/sdd/cor_fm.yml --data_dir data/sdd \
  --rotate --rotate_time_frame 6 --tied_noise --fm_in_scaling \
  --use_vqvae --fix_random_seed --seed 42 \
  --checkpt_freq 1 --exp teacher

python fm_giga.py --cfg cfg/giga/cor_fm.yml --data_dir data/giga \
  --data_source TUTR --subset giga \
  --rotate --rotate_time_frame 6 --tied_noise --fm_in_scaling \
  --use_vqvae --fix_random_seed --seed 42 --checkpt_freq 1 --exp teacher
```

For ETH-UCY, train the five leave-one-out folds separately by changing
`--subset` to `eth`, `hotel`, `univ`, `zara1`, and `zara2`, using the
corresponding prepared train/test files.
The default optimizer is AdamW, with learning rate `1e-4`, weight decay
`0.01`, and 150 epochs. `--epochs` and `--batch_size` override the YAML values.
Training writes checkpoints and an effective `cor_fm_updated.yml` inside the
run directory under `outputs/`. A Git checkout is not required.

The supplied YAML files have distinct purposes:

| Configuration | Training lengths | PTR codebook / code dimension | Encoder / decoder MoE |
|---|---|---|---|
| ETH-UCY | history `{2,4,6,8}`, future `{4,8,12}` | 64 / 128 | enabled / enabled |
| SDD | history `{2,4,6,8}`, future `{4,8,12}` | 64 / 128 | enabled / enabled |
| GigaTraj | history `8`, future `{30,60,90}` | 64 / 256 | disabled / enabled |

The GigaTraj configuration is taken from the fixed checkpoint used in the
dense horizon experiment, rather than an intermediate codebook sweep.
These are training entry-point configurations, not a claim that all paper
tables were produced with identical hyperparameters. Always evaluate a
checkpoint using its own saved effective configuration. A fresh training run
is not guaranteed to reproduce a particular existing checkpoint exactly.

## Fixed-checkpoint evaluation

The following examples use a saved teacher run's effective configuration
and `models/checkpoint_best.pt`. Replace `RUN` with that run's directory.
`evaluate_teacher.py` strictly loads the checkpoint's online `model` weights,
matching the original teacher evaluation. It does not substitute EMA weights.

```bash
python evaluate_teacher.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/eth_ucy --subset eth \
  --histories 2,4,6,8 --horizons 3,6,9,12 \
  --output outputs/eth_teacher_eval

python evaluate_teacher.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/sdd --histories 2,4,6,8 --horizons 3,6,9,12 \
  --output outputs/sdd_teacher_eval

python evaluate_teacher.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/giga --histories 4,6,8 --horizons 30:90 \
  --output outputs/giga_teacher_eval
```

Defaults are 20 proposals (from the checkpoint configuration), 100
`lin_poly` steps, polynomial degree 5, long-step parameter 1000, batch size
100, seed 42, and float32. Record these choices when comparing results.
Use `--check-only` to verify data, masks, and strict checkpoint loading on
CPU without sampling. `--device cpu` also permits actual CPU evaluation,
although GPU evaluation is recommended for the complete test set.

Every requested history retains the most recent frames in the original
8-frame input slots. Older slots are zeroed and the forecast origin is held
fixed. Each history generates a maximum-length forecast once. For endpoint
`Tf`, ADE averages its first `Tf` frames, while FDE uses frame `Tf`.
Each metric is independently minimized over proposals for each agent and
then averaged across the complete test set. Thus GigaTraj evaluates 183
length combinations for histories `{4,6,8}` and integer endpoints `30..90`.
This is prefix evaluation within the trained output capacity.

The evaluator writes `metrics.csv` with full-precision ADE/FDE and
`protocol.json` with sampling settings and checkpoint/configuration hashes.
Coordinates use the original preprocessed dataset units; no additional
unit conversion is applied. Use a fresh output directory to avoid mixing
different runs. Evaluation requires externally supplied data and weights.

The original single-configuration evaluator is also retained. For example:

```bash
python eval_giga.py --ckpt_path RUN/models/checkpoint_best.pt \
  --data_dir data/giga --data_source TUTR --subset giga \
  --rotate --rotate_time_frame 6 --fix_random_seed --seed 42 \
  --batch_size 100 --sampling_steps 100 --solver lin_poly \
  --lin_poly_p 5 --lin_poly_long_step 1000
```

## Packaging notes

Only source files, configurations, requirements, documentation, and
validation metadata are shipped. Personal paths and logging identifiers
have been removed. The numerical teacher model, MoE, PTR losses, sampler,
optimizer, and training loop retain their original implementations.
Student classes and loader branches have been removed; the cleaned loader
behavior is checked against the original teacher paths. The source-only
backup helper also works after extracting this archive without `.git`.
Evaluation timing uses CUDA events on GPU and a wall clock on CPU; forecasting
and metric calculations are unchanged.

See `THIRD_PARTY_NOTICES.md` and `LICENSE` for upstream attribution.

## Acknowledgments

We thank the authors of **MoFlow** for open-sourcing their code, which provides
the foundation for our implementation.
