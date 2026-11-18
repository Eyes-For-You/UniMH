# UniMH

## Abstract

Trajectory prediction is fundamental to applications such as autonomous driving, robot
navigation, and intelligent video surveillance, yet most existing methods are trained and
evaluated under a fixed observation–prediction horizon pair. In practice, the available
observation history and required prediction range vary across scenarios. We formulate *unified
multi-horizon trajectory prediction*, where a single model jointly handles multiple
observation–prediction horizon pairs without configuration-specific retraining. This task
requires modeling horizon-dependent motion distributions while preserving reusable trajectory
structures across configurations.To this end, we propose **UniMH**, a Transformer-based framework with two complementary
components. Motion Distribution Specialization (MDS) replaces standard feed-forward networks
with horizon-aware sparse expert routing, enabling feature transformations to adapt to local
motion context and temporal requirements. Prototype-Guided Trajectory Regularization (PTR)
introduces a training-only VQ-VAE branch with a shared learnable codebook, which regularizes
predicted trajectories through reconstruction and quantization objectives.Experiments on ETH-UCY, SDD, and GigaTraj demonstrate that UniMH achieves strong performance
using one shared checkpoint across diverse horizon configurations. Its advantages are
particularly consistent at long prediction horizons, where trajectory uncertainty and
accumulated drift are most pronounced.

## Installation

Use Python 3.11 and run all commands from the project root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

Set `DATA_ROOT` to the directory containing the preprocessed datasets.

```bash
DATA_ROOT=/path/to/preprocessed
mkdir -p data/eth_ucy data/sdd data/giga
cp -R "$DATA_ROOT/eth_ucy/original" data/eth_ucy/
cp -R "$DATA_ROOT/sdd/original" data/sdd/
cp -R "$DATA_ROOT/giga/TUTR" data/giga/
```

## Training

Train one model per dataset split:

```bash
python train_unimh.py --dataset eth_ucy --fold eth
python train_unimh.py --dataset sdd
python train_unimh.py --dataset giga
```

For ETH-UCY, repeat training and evaluation for `eth`, `hotel`, `univ`, `zara1`,
and `zara2`, then average the five scene results.

## Evaluation

For each dataset or fold, replace `RUN` with its training output directory under
`outputs/<dataset>/cor_fm/`. Use a separate evaluation output directory for each fold
or rerun. Evaluation uses CUDA by default; add `--device cpu` to run on CPU.

```bash
python evaluate.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/eth_ucy --subset eth \
  --histories 2,4,6,8 --horizons 3,6,9,12 \
  --output outputs/eth_eval

python evaluate.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/sdd --histories 2,4,6,8 --horizons 3,6,9,12 \
  --output outputs/sdd_eval

python evaluate.py \
  --cfg RUN/cor_fm_updated.yml \
  --checkpoint RUN/models/checkpoint_best.pt \
  --data-dir data/giga --histories 4,6,8 --horizons 30:90 \
  --output outputs/giga_eval
```

## Acknowledgments

We thank the authors of **MoFlow** for releasing their code; see [LICENSE](LICENSE).
