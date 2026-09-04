
# mmRehab

This repository contains the implementation of **mmRehab**, a model for estimating 3D human meshes from COTS mmWave radar during dynamic and static rehabilitation activities.

## Repository layout

- `code/main.py`: training and evaluation entry point.
- `code/run.py`: experiment launcher.
- `code/feature_extractors/mmrehab_kd.py`: mmRehab backbone and knowledge-distillation model.
- `code/trainer/`: training and evaluation loops.
- `code/signal_processing/`: radar parsing and signal-processing utilities.
- `code/if_generation/`: synthetic intermediate-frequency signal generation.
- `code/preprocessing/`: preprocessing utilities.
- `exps/main/config.json`: example experiment configuration.

## Environment

The code was prepared for Python 3.9, PyTorch, and CUDA 11.8. Create an environment and install PyTorch for your CUDA platform first:

```bash
conda create -n mmrehab python=3.9
conda activate mmrehab
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Build the PointNet++ CUDA extension:

```bash
cd code/modules/pointnet2
python setup.py install
cd ../../..
```

FFmpeg is required only for video generation.

## SMPL setup

SMPL model files are excluded because the official model license does not permit redistribution. Register and download the models from the [official SMPL website](https://smpl.is.tue.mpg.de/), then follow [`code/smpl/models/README.md`](code/smpl/models/README.md).

SMPL-X support is optional. It requires separately obtained SMPL-X model files and the `human_body_prior` package; both are governed by their own licenses.

## Dataset layout

Set `dataset.dir` in `exps/main/config.json` to the dataset root. Each configured dataset path is expected to contain subject and pose directories. A pose directory stores one folder per input or label field, with matching frame files:

```text
dataset_root/
└── configured_set/
    └── subject/
        └── pose/
            ├── range_azimuths/frame_1.npy
            ├── range_elevations/frame_1.npy
            ├── range_dopplers/frame_1.npy
            ├── smpl/frame_1.npy
            └── depth_mesh/frame_1.npy
```

The example configuration uses a teacher checkpoint at `checkpoints/teacher.pt`. Change `model.teacher.path` if the checkpoint is stored elsewhere.

## Training

Run commands from the `code` directory so the relative paths in the experiment configuration resolve correctly:

```bash
cd code
python run.py --exp_path ../exps --exp main --training
```

Training logs, TensorBoard events, and model checkpoints are written below `exps/main/`.

## Evaluation

```bash
cd code
python run.py --exp_path ../exps --exp main
```

The evaluator loads `exps/main/saved_models/model.pt` unless `model.pretrained_path` is set in the configuration.
