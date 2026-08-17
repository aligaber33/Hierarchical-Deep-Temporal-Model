# Hierarchical Deep Temporal Model — Volleyball Group Activity Recognition

A PyTorch re-implementation of the CVPR 2016 paper **"A Hierarchical Deep Temporal Model for Group Activity Recognition"** (Ibrahim et al.), applied to the **Volleyball Dataset**. The core idea is to recognize a *group activity* (e.g. `l_spike`, `r_set`, `l_winpoint`) from a short video clip by first modeling each **person's** individual action over time, then aggregating those person-level representations into a **group-level** temporal representation.

This repository contains a progression of models — from a simple single-frame image classifier up to a full two-stage, two-LSTM hierarchical model — plus the data-loading and feature-extraction utilities needed to train them on the Volleyball Dataset.

> 📄 Reference paper: [Deep Hierarchical Temporal Model for Group Activity Recognition (CVPR 2016)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2fSZbmkAAAAJ&citation_for_view=2fSZbmkAAAAJ:kNdYIx-mwKoC) — a copy is also included in this repo as `Implementing CVPR16 "Deep hierarchical temporal model for group recognition".pdf`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Model Progression (B1 → B7)](#model-progression-b1--b7)
- [Requirements & Installation](#requirements--installation)
- [Dataset Setup](#dataset-setup)
- [Pretrained Weights](#pretrained-weights)
- [Usage](#usage)
  - [1. Parsing annotations](#1-parsing-annotations)
  - [2. Extracting CNN features](#2-extracting-cnn-features)
  - [3. Training a baseline](#3-training-a-baseline)
  - [4. Loading checkpoints (single or multi-GPU)](#4-loading-checkpoints-single-or-multi-gpu)
- [Configuration Notes](#configuration-notes)
- [Known Issues / Caveats](#known-issues--caveats)
- [Citation](#citation)

---

## Project Overview

Group activity recognition asks: *given several people interacting in a scene over time, what is the group as a whole doing?* The paper (and this codebase) approaches this hierarchically:

1. **Person level** — a CNN (ResNet-50) extracts appearance features for each detected/tracked player in every frame of a clip.
2. **Person-temporal level** — an LSTM consumes each player's per-frame features over time to build a temporal representation of that individual's action.
3. **Group level** — the individual player representations are pooled (max-pooling) across all players in a frame/clip, and optionally passed through a second LSTM to model how the *pooled group representation* evolves over time.
4. A final classifier head maps the group representation to one of the **group activity classes**.

The repo implements this idea incrementally as a series of "baselines" (`B1`–`B6`) that each add one more piece of the hierarchy, culminating in the full two-stage model (`B7`, in `Full_Model.py`).

---

## Repository Structure

The repository is organized into two top-level folders that group the training scripts and the saved weights, respectively:

```
Hierarchical-Deep-Temporal-Model-main/
│
├── Source Code/                # All B1–B7 model/training scripts, plus their shared dependencies
│   ├── Baseline_B1.py          # B1: single mid-frame image classifier (ResNet-50, fine-tuned)
│   ├── Baseline_B2.py          # B2: MLP classifier on precomputed per-person ResNet features
│   ├── Baseline_B3.py          # B3: pooled per-person features (avg over time) -> MLP classifier
│   ├── Baseline_b4.py          # B4: end-to-end CNN + single LSTM over full clip frames (image-level)
│   ├── Baseline_b5.py          # B5: end-to-end CNN + per-player crops + LSTM + max-pool -> classifier
│   ├── Baseline_B6.py          # B6: same as B5 but with partial backbone fine-tuning + presence mask
│   ├── Full_Model.py           # B7 / TwoStageModelB7: full hierarchical model (player LSTM + group LSTM)
│   ├── Box_info.py             # Parses a single tracking-annotation line into a BoxInfo object
│   ├── Annotation_loader.py    # Loads/aggregates Volleyball dataset tracking annotations
│   └── extract_feat.py         # Extracts ResNet-50 CNN features (image-level or person-level)
│
├── Models Weights/              # All saved checkpoints (Git LFS pointers — see below)
│   ├── Pytorch_baseline_b1.pth
│   ├── Pytorch_baseline_b2.pth
│   ├── Pytorch_baseline_b3.pth
│   ├── Pytorch_baseline_b4.pth
│   ├── Pytorch_baseline_b5.pth
│   ├── Pytorch_baseline_b6.pth
│   └── Final_model_weights.pth  # Full B7 / TwoStageModelB7 weights
│
├── Box_info.py                 # (root copy) kept for backward-compat / standalone use
├── Annotation_loader.py        # (root copy) kept for backward-compat / standalone use
├── extract_feat.py             # (root copy) kept for backward-compat / standalone use
│
├── features/                   # Precomputed CNN feature caches (.npy)
│   ├── image-level/resnet/     # One feature vector per frame (whole image)
│   └── person-level/resnet/    # One feature vector per detected player per frame
│
├── debug_output.png            # Sample debug visualization
├── Implementing CVPR16 ....pdf # Copy of the reference paper
│
├── .lightning_studio/          # Lightning AI Studio startup/shutdown hooks (on_start.sh / on_stop.sh)
├── .vscode/                    # Editor settings (Lightning AI cloud studio defaults)
├── .gitattributes              # Marks *.pth files as Git LFS objects
└── .gitignore                  # Ignores the raw video/annotation folders (not checked into git)
```

> **Note on `Source Code/`:** `Box_info.py`, `Annotation_loader.py`, and `extract_feat.py` are not "B1–B7" models themselves, but `Baseline_B1.py` (imports `extract_feat`), `Baseline_b4.py`/`Baseline_b5.py`/`Baseline_B6.py`/`Full_Model.py` (import `Baseline_B1`/`Baseline_B3`/`Baseline_B6`), and `Annotation_loader.py` (imports `Box_info`) all depend on each other via plain top-level `import`/`from ... import ...` statements — not package-relative imports. Python resolves these against the directory the entry-point script lives in, so a copy of these three helper files is kept **inside** `Source Code/` alongside the baselines to keep every B1–B7 script runnable on its own without editing any import statements. The original copies remain at the repo root too, unchanged, for anything that still references them there (e.g. standalone feature extraction).

Two folders referenced heavily throughout the code are **not included in the repo** (they're in `.gitignore` because they're large raw datasets) and must be downloaded/placed manually — see [Dataset Setup](#dataset-setup):

- `videos-splitted/` — the raw Volleyball Dataset video frames.
- `volleyball_tracking_annotation/` — the per-player tracking annotation `.txt` files.

---

## Model Progression (B1 → B7)

| Script (in `Source Code/`) | Model | Input | Temporal modeling | Notes |
|---|---|---|---|---|
| `Baseline_B1.py` | `Image_Classifier` | One mid-clip RGB frame | None | Fine-tuned ResNet-50 with a custom MLP head. Trains/evaluates directly on raw images (`videos-splitted/`). |
| `Baseline_B2.py` | `PersonClassifier` | Precomputed per-person ResNet features | None (per-frame, per-person classification) | MLP classifier for **individual player action** (9 classes: waiting, setting, digging, falling, spiking, blocking, jumping, moving, standing), not group activity. |
| `Baseline_B3.py` | `GroupActivityRecognition` | Precomputed person-level features, shape `[frames, 12 players, 2048]` | Mean-pooled over time (no LSTM) | Flattens 12×2048 pooled features into an MLP for group activity (8 classes). |
| `Baseline_b4.py` | `TemporalClassifier` | Raw frame sequence (whole image, no player crops) | Single-layer LSTM over frame features | End-to-end CNN+LSTM baseline; does not model individual players. |
| `Baseline_b5.py` | `TemporalModelB5` | Per-player crop sequence `[B, T, N, C, H, W]` | Single LSTM over pooled (max over players) frame representations | First model to use per-player crops; player features are max-pooled *before* the LSTM. |
| `Baseline_B6.py` | `TwoStageModelB6` | Per-player crop sequence + presence mask | One LSTM per player (`player_lstm`), then max-pool over players | Backbone is partially frozen (only `layer4` is fine-tuned). Saves full training checkpoints (`model_state_dict` + `optimizer_state_dict` + `epoch`). |
| `Full_Model.py` | `TwoStageModelB7` (this is "B7") | Per-player crop sequence + presence mask | **Two-stage**: per-player LSTM over time → max-pool over players per frame → second LSTM over the pooled group sequence | The full hierarchical model described in the paper: person-temporal stage + group-temporal stage. |

All group-activity models classify into the 8 Volleyball Dataset group classes:

```python
categories_dct = {
    'l_pass': 0, 'r_pass': 1, 'l_spike': 2, 'r_spike': 3,
    'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
}
```

---

## Requirements & Installation

The repo has no `requirements.txt` / `environment.yml`; dependencies were inferred from the imports used across all scripts. It was developed/run on a **Lightning AI Studio** cloud workspace (see `.lightning_studio/`, `.vscode/settings.json`) with Python 3.12.

Install the dependencies with:

```bash
pip install torch torchvision numpy opencv-python pillow
```

| Package | Used for |
|---|---|
| `torch`, `torchvision` | Models (ResNet-50 backbones, LSTMs), datasets/dataloaders, image transforms |
| `numpy` | Feature array storage (`.npy`) and manipulation |
| `opencv-python` (`cv2`) | Drawing debug bounding boxes and writing debug videos in `Annotation_loader.py` |
| `pillow` (`PIL`) | Image loading/cropping for datasets and feature extraction |

**GPU:** A CUDA-capable GPU is strongly recommended — all scripts auto-detect it via:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Some checkpoints in this repo (e.g. `Final_model_weights.pth`, `Pytorch_baseline_b6.pth`) were produced from training runs on **2 GPUs**. See [Loading checkpoints](#4-loading-checkpoints-single-or-multi-gpu) for how to load these correctly regardless of how many GPUs you have available at inference time.

**Git LFS:** The `.pth` checkpoint files are tracked via Git LFS (`.gitattributes`). Make sure you have [Git LFS](https://git-lfs.github.com/) installed and run `git lfs pull` after cloning, otherwise you'll only get small pointer files instead of the actual weights.

---

## Dataset Setup

This project expects the **Volleyball Dataset** (Ibrahim et al., CVPR 2016) laid out as two sibling folders at the repository root:

```
videos-splitted/
├── videos_g1/
│   ├── <video_id>/
│   │   └── <clip_id>/
│   │       ├── <frame_id>.jpg
│   │       └── annotations.txt      # per-clip group-activity label
├── videos_g2/
│   └── ...
└── ...

volleyball_tracking_annotation/
├── <video_id>/
│   └── <clip_id>/
│       └── <clip_id>.txt            # per-frame, per-player tracking boxes + individual action
```

- `annotations.txt` (inside each `videos_g*/<video_id>/` folder) maps a clip's middle-frame filename to its **group activity** label (e.g. `l_spike`, `r_winpoint`).
- `<clip_id>.txt` (inside `volleyball_tracking_annotation/`) contains one line per tracked player per frame, parsed by `Box_info.py`:

  ```
  player_ID x1 y1 x2 y2 frame_ID lost grouping generated action_label
  ```

Both folders are listed in `.gitignore` and are **not included in this repository** — download the official Volleyball Dataset separately and place it at the repo root using the layout above (some scripts use relative paths like `Path("videos-splitted")`, others use absolute Lightning Studio paths — see [Configuration Notes](#configuration-notes)).

---

## Pretrained Weights

All checkpoints live under `Models Weights/`:

| File | Corresponds to | Size (real weights, via Git LFS) |
|---|---|---|
| `Models Weights/Pytorch_baseline_b1.pth` | `Baseline_B1.Image_Classifier` | ~94 MB |
| `Models Weights/Pytorch_baseline_b2.pth` | `Baseline_B2.PersonClassifier` | ~1 MB |
| `Models Weights/Pytorch_baseline_b3.pth` | `Baseline_B3.GroupActivityRecognition` | ~13 MB |
| `Models Weights/Pytorch_baseline_b4.pth` | `Baseline_b4.TemporalClassifier` | ~95 MB |
| `Models Weights/Pytorch_baseline_b5.pth` | `Baseline_b5.TemporalModelB5` | ~97 MB |
| `Models Weights/Pytorch_baseline_b6.pth` | `Baseline_B6.TwoStageModelB6` (raw checkpoint dict — see below) | ~97 MB |
| `Models Weights/Final_model_weights.pth` | `Full_Model.TwoStageModelB7` (B7) | ~118 MB |

⚠️ **Two checkpoint formats exist in this repo:**
- `Baseline_B1.py`–`Baseline_b5.py` save a **plain `state_dict()`**:
  ```python
  torch.save(model.state_dict(), "Pytorch_baseline_b1.pth")
  ```
- `Baseline_B6.py` and `Full_Model.py` save a **full training checkpoint dict**:
  ```python
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': running_loss,
  }
  torch.save(checkpoint, checkpoint_path)
  ```
  So loading `B6_checkpoint.pth`-style files requires indexing `checkpoint['model_state_dict']` first. The generic loader below handles both cases automatically.

---

## Usage

### 1. Parsing annotations

`Annotation_loader.py` builds an in-memory (and optionally pickled) index of the dataset, and can visualize a clip's tracked bounding boxes as a debug video. Run it from the repository root (so relative data paths like `videos-splitted/` resolve correctly):

```bash
python "Source Code/Annotation_loader.py"
```

By default, running the file directly visualizes a single hard-coded clip (`video_id="4"`, `clip_id="24745"`) and writes `output_clip.mp4`. To build a full pickled annotation index instead, call:

```python
from Annotation_loader import create_pkl_version
create_pkl_version()   # writes annot_all.pkl
```

### 2. Extracting CNN features

`extract_feat.py` runs a frozen ResNet-50 (ImageNet-pretrained) over the dataset and caches feature vectors as `.npy` files, so downstream baselines (B2/B3) don't need to re-run the CNN every epoch.

```bash
python "Source Code/extract_feat.py"
```

Inside the script, toggle the extraction mode:

```python
image_level = True   # True  -> one 2048-d feature vector per frame  (features/image-level/resnet/...)
                      # False -> one [12, 2048] feature matrix per frame (features/person-level/resnet/...)
```

Output shape per clip is `[num_frames, num_people, 2048]` (with `num_people = 1` for image-level), zero-padded to a fixed `9` frames and `12` people.

### 3. Training a baseline

Each `Baseline_*.py` / `Full_Model.py` file (in `Source Code/`) is self-contained and runnable as a script — it builds its dataset, trains for a fixed number of epochs, evaluates on a held-out split, and saves a checkpoint. **Run these from the repository root** so relative paths (`videos-splitted/`, `features/...`) still resolve correctly:

```bash
# Frame-level classifier (no precomputed features needed, reads videos-splitted/ directly)
python "Source Code/Baseline_B1.py"

# Per-person MLP on precomputed person-level features
python "Source Code/Baseline_B2.py"

# Pooled-features MLP for group activity
python "Source Code/Baseline_B3.py"

# End-to-end CNN+LSTM over whole frames
python "Source Code/Baseline_b4.py"

# End-to-end CNN+LSTM over per-player crops
python "Source Code/Baseline_b5.py"

# Two-stage model with partial fine-tuning + presence mask (saves full checkpoints each epoch)
python "Source Code/Baseline_B6.py"

# Full hierarchical two-stage model (player LSTM -> group LSTM) = "B7"
python "Source Code/Full_Model.py"
```

Each script saves its output checkpoint (`.pth`) into the **current working directory** by default (i.e. the repo root, if you followed the command above) — move/copy new checkpoints into `Models Weights/` yourself if you want them to live alongside the existing pretrained ones.

Each script prints per-epoch training loss/accuracy and a final test-set accuracy. Hyperparameters (epochs, batch size, learning rate, LSTM hidden size, etc.) are set as constants near the top of each `__main__` block — edit them directly in the script to change them, since there is no CLI/config file.

### 4. Loading checkpoints (single or multi-GPU)

Because some checkpoints in this repo were saved from a **`DataParallel`/multi-GPU** training run (which prefixes every parameter name with `module.`) and others were saved from a single-GPU/CPU run (no prefix), naively calling `model.load_state_dict(torch.load(path))` can fail with `Missing key(s) in state_dict` / `Unexpected key(s) in state_dict` errors depending on which environment you're loading in.

Use a generalized loader that strips (or restores) the `module.` prefix as needed and works regardless of how many GPUs are available at load time:

```python
import torch
import torch.nn as nn


def load_checkpoint(model, checkpoint_path, device=None, optimizer=None):
    """
    Loads a model checkpoint saved from either a single-GPU/CPU run or a
    multi-GPU (nn.DataParallel / DistributedDataParallel) run, onto any
    target device setup (CPU, 1 GPU, or multiple GPUs).

    Handles both checkpoint formats used in this repo:
      - a plain state_dict (Baseline_B1..B5)
      - a full dict with 'model_state_dict' / 'optimizer_state_dict' / 'epoch' (Baseline_B6, Full_Model)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint_path, map_location=device)

    # Unwrap full training checkpoints (B6 / B7 style)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw

    # Normalize 'module.' prefixes so it doesn't matter whether the checkpoint
    # was saved from nn.DataParallel (multi-GPU) or a plain model (single-GPU/CPU)
    is_parallel_model = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if has_module_prefix and not is_parallel_model:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and is_parallel_model:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    if optimizer is not None and isinstance(raw, dict) and "optimizer_state_dict" in raw:
        optimizer.load_state_dict(raw["optimizer_state_dict"])

    epoch = raw.get("epoch") if isinstance(raw, dict) else None
    return model, epoch


# --- Example usage ---
# Run this from the repository root, e.g.: python "Source Code/your_script.py"
import sys
sys.path.insert(0, "Source Code")   # so `from Full_Model import ...` resolves
from Full_Model import TwoStageModelB7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoStageModelB7()

# Optionally wrap in DataParallel if you have multiple GPUs available now
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model, epoch = load_checkpoint(model, "Models Weights/Final_model_weights.pth", device=device)
model.eval()
```

This makes weight loading independent of whether:
- the checkpoint was originally saved from 1 GPU or 2 GPUs, and
- you are currently loading it onto a CPU, a single GPU, or multiple GPUs.

---

## Configuration Notes

- **Run scripts from the repository root**, not from inside `Source Code/` — e.g. `python "Source Code/Baseline_B1.py"`. All relative paths in the code (`videos-splitted/`, `features/...`) are resolved against the current working directory, so invoking from the root keeps them pointed at the top-level `videos-splitted/`, `volleyball_tracking_annotation/`, and `features/` folders.
- **Hard-coded paths differ between scripts.** `Baseline_B1.py`–`Baseline_b5.py` use relative paths (`Path("videos-splitted")`, `Path("features/...")`), while `Baseline_B6.py` and `Full_Model.py` use absolute Lightning AI Studio paths (`/teamspace/studios/this_studio/videos-splitted`). If you're not running inside that same Lightning Studio environment, edit `vids_root` / `annot_root` at the top of those two files (in `Source Code/`) to match your local dataset location.
- **No CLI arguments / config file** — all hyperparameters (epoch count, batch size, learning rate, crop size, max players, target frames, etc.) are Python constants inside each script. Edit the source directly to change training settings.
- **`.lightning_studio/on_start.sh` / `on_stop.sh`** are empty hooks for the Lightning AI cloud IDE (run on studio start/stop) — safe to ignore if running elsewhere.

## Citation

If you use this implementation, please cite the original paper:

```
Ibrahim, M. S., Muralidharan, S., Deng, Z., Vahdat, A., & Mori, G. (2016).
A Hierarchical Deep Temporal Model for Group Activity Recognition.
CVPR 2016.
```
