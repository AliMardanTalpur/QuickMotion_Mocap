# Tracked-crop to SMPL pipeline

This is a separate path from the earlier VideoPose3D/BVH/FBX scripts. The
three commands run in separate processes. Stage 1 uses YOLOv8 pose plus
ByteTrack; Stage 2 uses the **official 4DHumans HMR2** checkpoint on one saved
crop at a time; Stage 3 filters SMPL rotations on SO(3) and makes conservative
planted-foot root corrections. The result is a pickle-free NPZ for later Unity
retargeting, **not an FBX**.

## Current status and limits

- Stage 1 ran on `dance.mp4`: 154 of 155 frames have the same tracked actor ID.
- The HMR2 model itself has **not** run on this machine. Its checkpoint, official
  repository dependencies, and licensed SMPL neutral model are not installed.
  The current `.venv` has CPU-only PyTorch and Python 3.13. Use a separate
  environment supported by the official 4DHumans repository for Stage 2.
- `--batch-size 1` is enforced. The CUDA process sets a 3072 MB PyTorch
  allocator limit and records its peak reserved memory. This is an operational
  gate, not proof that HMR2 fits in 3 GB on an RTX 2050. Select `--device cpu`
  if model loading or inference exceeds the limit.
- HMR2 estimates SMPL pose **in camera space**. Camera-derived translation and
  ground/contact cleanup are approximate; this path does not claim measured
  world trajectory or guaranteed zero foot sliding. Stage 3 estimates one
  support plane and shifts it to zero; its NPZ metadata records
  `ground_aligned: estimated` and `world_trajectory_is_estimate: true`.
- Inference makes no network requests. Acquire the official checkpoint and
  licensed SMPL neutral model during setup. The model files cannot be supplied
  by this project.

## Setup

For Stages 1 and 3 in this repository:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For Stage 2, follow the [official 4DHumans installation instructions](https://github.com/shubham-goel/4D-Humans#installation-and-setup)
in a separate compatible Python environment. Obtain the official HMR2
checkpoint and the licensed SMPL neutral body model as documented there. The
SMPL neutral model must be at one of the locations accepted by 4DHumans,
including `<4DHumans>/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`.
Run the Stage 2 command with **that environment's Python**, not necessarily
this project's `.venv` Python. A WSL2 Linux environment may be easier for
4DHumans dependencies than native Windows; the same `runs/` files can be read
from the mounted project directory.

## Run

Stage 1 is ready with the existing `.venv`:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\.venv\Scripts\python.exe commands\01_track.py --video dance.mp4 --run runs\dance_arch --model yolov8n-pose.pt --device cpu --vram-limit-mb 3072
```

This writes `runs/dance_arch/tracking.npz`, a JSON metadata sidecar, and
`runs/dance_arch/crops/000000.png` etc. Crops are 512-pixel squares by
default; `crop_to_image` in the NPZ reverses the crop transform. Missing
observations remain flagged even if a short gap receives an interpolated crop.
The current project `.venv` has CPU-only PyTorch. To use the RTX 2050 for
Stage 1, first install a CUDA-enabled PyTorch wheel using the
[official selector](https://pytorch.org/get-started/locally/), then use
`--device 0`; Stage 1 and Stage 2 still run as separate processes.

Stage 2 needs real, existing paths. Replace the two example paths below and
run with the **4DHumans environment's Python**:

```powershell
<HMR2_PYTHON> commands\02_solve.py --run runs\dance_arch --solver hmr2 --hmr2-repo '<4DHumans checkout>' --checkpoint '<HMR2 .ckpt file>' --batch-size 1 --device cpu --vram-limit-mb 3072
```

Once the model runs on CPU and you have a CUDA-enabled PyTorch in that
environment, change `--device cpu` to `--device cuda`. If the 3 GB gate fails,
keep the CPU setting. The output is `runs/dance_arch/raw_smpl.npz`.

Stage 3 uses this project's `.venv` after Stage 2 succeeds:

```powershell
.\.venv\Scripts\python.exe commands\03_filter_export.py --run runs\dance_arch --output runs\dance_arch\motion_unity.npz --foot-lock on
```

`motion_unity.npz` contains `rotations` `(N,24,3,3)`, one stable `betas`
`(10,)`, `translation` `(N,3)`, predicted `joints` `(N,24,3)`, `valid`, and
`confidence`. The JSON sidecar states FPS, model, units, coordinate assumptions,
and validity semantics. `quality.json` records contact corrections and the
solver's peak PyTorch reserved VRAM. The `joints` array is a diagnostic from
the raw model with corrected root translation; a Unity retargeter should use
the rotation matrices and translation and then recompute joints via its rig.

## Check without model files

The geometry/contract tests require no 4DHumans model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
