# GVHMR route for `dance.mp4`

This is the higher quality motion recovery route for clips that the COCO-17 +
VideoPose3D pipeline cannot reconstruct well. GVHMR estimates articulated
SMPL-X motion before this project's Blender skeleton export. It runs on CPU in
its own Python environment to stay within the RTX 2050's 4 GB VRAM. The source
video is 25 FPS; GVHMR stages it at its trained 30 FPS rate, so the exported
motion will contain about 186 frames rather than 155.

The `GVHMR-portable` checkout is the maintained Windows-capable port at commit
`e3876097a9c4d9c1758d058d9e70c64f0cdba4c4`. Its research license and
the registered SMPL-X body model's license apply. The licensed body model and
checkpoints stay in the ignored local checkout; do not commit or redistribute
them.

## Run from the current project folder

Open PowerShell and run:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\GVHMR-portable\.venv\Scripts\python.exe gvhmr_infer.py --preflight
.\GVHMR-portable\.venv\Scripts\python.exe gvhmr_infer.py --video dance.mp4 --repo GVHMR-portable --output-dir runs\dance_gvhmr --flip-test
.\.venv\Scripts\python.exe gvhmr_refine.py --run-dir runs\dance_gvhmr --output runs\dance_gvhmr\smplx24_refined_v2.npy
.\.venv\Scripts\python.exe motion_preview.py --input runs\dance_gvhmr\smplx24_refined_v2.npy --output runs\dance_gvhmr\motion_preview_refined_v2.jpg --samples 16
.\.venv\Scripts\python.exe gvhmr_review.py --run-dir runs\dance_gvhmr --joints runs\dance_gvhmr\smplx24_refined_v2.npy --output runs\dance_gvhmr\review_overlay_refined_v2.jpg --samples 16
.\.venv\Scripts\python.exe stage3_export.py --input runs\dance_gvhmr\smplx24_refined_v2.npy --fps 30 --bvh runs\dance_gvhmr\dance_gvhmr_refined_v2.bvh --fbx runs\dance_gvhmr\dance_gvhmr_refined_v2.fbx --ground fixed --foot-lock off
```

Import `runs\dance_gvhmr\dance_gvhmr_refined_v2.fbx` in Blender with **File > Import >
FBX**. The first Python command checks the registered SMPL-X model and all
four public checkpoints. If it lists missing checkpoints, download them into
the fork's configured checkpoint directory:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP\GVHMR-portable'
.\.venv\Scripts\python.exe -c "from gvhmr.utils.assets import fetch,select; print(fetch(select(group='demo')))"
cd ..
```

The public checkpoints total about 5.6 GB and may take time to download.
`SMPLX_NEUTRAL.npz` must come from your own registration at
<https://smpl-x.is.tue.mpg.de/> and be placed at
`GVHMR-portable\inputs\checkpoints\body_models\smplx\SMPLX_NEUTRAL.npz`.
This checkout already has the licensed file you supplied.

## Files to inspect

- `runs\dance_gvhmr\gvhmr_params.npz`: the original recovered SMPL-X motion.
- `runs\dance_gvhmr\smplx24_camera.npy`: untouched camera-space SMPL-X joints.
- `runs\dance_gvhmr\smplx24_refined_v2.npy`: 24 joints after 2D-guided knee/ankle and elbow/wrist fitting.
- `runs\dance_gvhmr\3d_points_gvhmr.npy`: 17-joint bridge for comparison with the older pipeline.
- `runs\dance_gvhmr\motion_preview_refined_v2.jpg`: front and side diagnostic poses.
- `runs\dance_gvhmr\review_overlay_refined_v2.jpg`: projected 3D joints against the video and 2D detections.
- `runs\dance_gvhmr\smplx24_refined_v2.json`: limb fit, bone length, and reprojection metrics.
- `runs\dance_gvhmr\dance_gvhmr_refined_v2.quality.json`: Blender bake and FBX reimport errors.
- `runs\dance_gvhmr\dance_gvhmr_refined_v2.fbx` and `.bvh`: recommended animated skeletons.

The main export keeps GVHMR's body spine, collar, and foot joints. The fit
uses high-confidence ViTPose knees/ankles and elbows/wrists to correct large
visible misses, keeps GVHMR's depth as a prior, holds limb lengths fixed, and
smooths the correction across frames. The previous `dance_gvhmr_refined.fbx`
is still available for comparison. To compare against untouched GVHMR or the former
17-joint rig, run:

```powershell
.\.venv\Scripts\python.exe stage3_export.py --input runs\dance_gvhmr\smplx24_camera.npy --fps 30 --bvh runs\dance_gvhmr\dance_gvhmr.bvh --fbx runs\dance_gvhmr\dance_gvhmr.fbx --ground fixed --foot-lock off
.\.venv\Scripts\python.exe stage3_export.py --input runs\dance_gvhmr\3d_points_gvhmr.npy --bvh runs\dance_gvhmr\dance_gvhmr_17.bvh --fbx runs\dance_gvhmr\dance_gvhmr_17.fbx --ground fixed --foot-lock off
```

The source is one camera. GVHMR uses a learned body and motion prior to infer
hidden limbs and depth; fully occluded motion cannot be confirmed from this
video alone. Review both views of the contact sheet and the full FBX playback.

For this 186-frame run, the left ankle's 95th percentile 2D reprojection
error improved from 304 to 48 pixels and the right ankle's from 186 to 74
pixels after refinement. The left/right wrist's 95th percentile error improved
from 84/90 to 37/37 pixels. The sampled full-body overlay's median/p95 error
improved from 23.5/61.2 to 20.0/46.2 pixels compared with the previous leg-only
result. The refined FBX reimported within 0.00002 m of the 24-joint targets
at the 95th percentile. These are alignment and export checks,
not proof of true hidden-side depth from a single camera.

## Fresh environment setup

If the ignored `GVHMR-portable` checkout or its environment is missing, restore
it first. The commands below keep all downloads in this project:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
git clone https://github.com/ryanrudes/gvhmr GVHMR-portable
git -C GVHMR-portable checkout e3876097a9c4d9c1758d058d9e70c64f0cdba4c4
.\.venv\Scripts\python.exe -m pip install uv
$env:UV_CACHE_DIR='C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP\.uv-cache'
.\.venv\Scripts\python.exe -m uv python install 3.13 --install-dir .uv-python
.\.venv\Scripts\python.exe -m uv venv GVHMR-portable\.venv --python .uv-python\cpython-3.13-windows-x86_64-none\python.exe
cd GVHMR-portable
..\.venv\Scripts\python.exe -m uv pip install --python .venv\Scripts\python.exe torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
..\.venv\Scripts\python.exe -m uv pip install --python .venv\Scripts\python.exe -e . ultralytics lapx yacs pycolmap
cd ..
```

The port's full `preproc` extra currently includes `cython-bbox`, which needs
a local MSVC C++ compiler on Windows. The runtime imports used by this route
work with the packages above. Place your registered SMPL-X model in the path
shown earlier, then run the public-checkpoint command and the run commands.
