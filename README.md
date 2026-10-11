# Video to animated skeleton

For the Figma-based local QuickMotion interface, see [UI_RUN.md](UI_RUN.md).
It uploads video, runs the GVHMR route, shows progress and 3D previews, and
downloads FBX/BVH without entering commands for every capture.

For the higher quality GVHMR route for `dance.mp4`, see
[GVHMR_RUN.md](GVHMR_RUN.md). It uses its own CPU environment and licensed
SMPL-X body model, then exports an animated FBX/BVH skeleton.

The new YOLO tracking → HMR2 SMPL → Unity NPZ architecture is documented in
[NEW_PIPELINE.md](NEW_PIPELINE.md). The commands below describe the older
VideoPose3D/BVH/FBX pipeline.

This pipeline processes one person in a prerecorded MP4. Each stage runs as a
separate process, so YOLOv8 pose and VideoPose3D never occupy GPU memory at the
same time. Blender runs the final stage on CPU. The output is an animated
armature in BVH and FBX formats, without a skinned mesh.

## Setup

Use a Python version with a compatible `bpy` wheel (Python 3.11 or 3.13 on
Windows), or run Stage 3 with a separately installed Blender executable. Install
the requirements in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For RTX 2050 acceleration, install the PyTorch CUDA wheel suited to the
installed NVIDIA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/).
If CUDA is unavailable, both inference stages can run on CPU. The `bpy` wheel
is large; when using the Blender executable, it can be omitted from the Stage
1/2 environment because Blender supplies `bpy` itself.

## Run

Run these commands in order. Stage 2 downloads the [official VideoPose3D COCO
checkpoint](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)
on first use (or pass `--checkpoint` with a local copy). The checkpoint is only
loaded after Stage 1 has exited.

```powershell
python stage1_2d_extract.py input.mp4 --output 2d_keypoints.npy --device auto
python stage2_3d_lift.py --input 2d_keypoints.npy --output 3d_points.npy --device auto --chunk 32
blender -b --factory-startup --python stage3_export.py -- --input 3d_points.npy --bvh motion.bvh --fbx motion.fbx
```

The `bpy` Python package also allows `python stage3_export.py --input
3d_points.npy --bvh motion.bvh --fbx motion.fbx`, provided the package and
export operators are available for that Python version. The Blender command is
the recommended headless route.

When importing `motion.bvh` into Blender, set **Forward = Y** and **Up = Z**.
Blender's BVH exporter writes its Z-up coordinate convention; Blender's BVH
importer defaults to a different convention and otherwise displays it sideways.
The FBX imports upright with Blender's default import settings.

Stage 1 uses YOLOv8n-pose at 960 pixels by default, then writes COCO keypoints
`(N,17,3)` as pixel X/Y/confidence plus a JSON sidecar with FPS and frame size.
Stage 2 interpolates missed joints, repairs isolated detections, and applies
the checkpoint's exact screen coordinate normalization. Its 243-frame model
uses 121 past and 121 future frames, with edge padding for short clips. It
also averages a sequential mirrored inference pass, smooths the 3D joints,
enforces stable bone lengths, applies a frontal-camera depth prior for ordinary
poses, and writes Human3.6M positions `(N,17,3)` with
an FPS/joint order sidecar. Stage 3 maps those joints to a connected Blender
skeleton, animates IK targets, grounds feet or hands as appropriate, bakes the evaluated pose,
detects likely planted ankles and gently corrects root travel to reduce foot
sliding, checks the baked joint errors, and exports both files plus
`motion.quality.json`. The quality JSON records contact drift before and after
correction. Use `--foot-lock off` when the feet deliberately slide (for example,
moonwalking) or when the contact detector mistakes fast steps for planted feet.
For more accurate 2D joints, use `--model yolov8s-pose.pt`; it downloads on
first use when network access is available. If inference exceeds 4 GB VRAM,
reduce `--imgsz` to 640.
The `--depth-prior front` default constrains the head and torso's uncertain
front-to-back displacement more than the limbs. It improves the side view of a
mostly frontal video without changing its camera-view pose much. Use
`--depth-prior model` to keep the VideoPose3D depth unchanged for footage shot
from the side or with substantial turns. `--depth-scale` is an additional
uniform depth adjustment; its default is `1.0`.

### Acrobatics and floor work

The Human3.6M-trained lifter can misread handstands, breakdance moves, and
other unusual poses. Stage 2 now measures how well its 3D result projects back
onto the detected 2D limbs. With `--pose-mode auto` (the default), a large
median mismatch selects image-guided reconstruction: measured COCO joints
drive the screen-plane pose, symmetric fixed bone lengths constrain depth,
and the temporal model supplies front/back depth hints. Stage 3 then considers
both wrists and ankles when placing the floor. Use `--pose-mode model` to force
the original lifter or `--pose-mode image-guided` to force the new path.

For the included `dance.mp4`, use the static-camera white-shoe recovery. It
checks bright moving shoe regions against the dark leg silhouette when YOLO
collapses both ankles onto one shoe. This option is specific to footage with
a fixed camera, white shoes, and a background that can be estimated from the
video. The `planar` depth prior limits implausible front/back limb displacement
in this mostly side-on clip; it still retains the depth required by rigid bone
lengths. Review the preview from both camera and side angles before using the
FBX in a production scene.

Run the complete dance pipeline with separate output names:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\.venv\Scripts\python.exe stage1_2d_extract.py dance.mp4 --output 2d_keypoints_dance_v5.npy --model yolov8n-pose.pt --imgsz 960 --recover-white-shoes --device cpu
.\.venv\Scripts\python.exe stage2_3d_lift.py --input 2d_keypoints_dance_v5.npy --output 3d_points_dance_v6.npy --pose-mode image-guided --depth-prior planar --device cpu
.\.venv\Scripts\python.exe stage3_export.py --input 3d_points_dance_v6.npy --bvh dance_v6.bvh --fbx dance_v6.fbx --ground auto --foot-lock off
.\.venv\Scripts\python.exe motion_preview.py --input 3d_points_dance_v6.npy --output dance_v6_3d_contact_sheet.jpg --samples 16
```

The resulting `dance_v6.fbx` is the file to import in Blender. The `.json`
sidecar reports the selected pose mode and source-model reprojection error.
The white-shoe correction runs on CPU after YOLO is unloaded. It cannot
recover joints hidden for long periods, and the planar depth prior is an
assumption about this video rather than measured depth. For a truly accurate
side view of spinning or self-occluded motion, use synchronized additional
cameras or manually correct ambiguous keyframes.

To recreate the separately named 30 FPS output used for the depth comparison:

```powershell
.\.venv\Scripts\python.exe stage1_2d_extract.py mocap30fps.mp4 --output 2d_keypoints_30fps.npy
.\.venv\Scripts\python.exe stage2_3d_lift.py --input 2d_keypoints_30fps.npy --output 3d_points_30fps_frontprior.npy --depth-prior front
.\.venv\Scripts\python.exe stage3_export.py --input 3d_points_30fps_frontprior.npy --bvh motion_30fps_frontprior.bvh --fbx motion_30fps_frontprior.fbx
```

For the 25 FPS clip, keep its outputs separate:

```powershell
.\.venv\Scripts\python.exe stage1_2d_extract.py mocap25fps.mp4 --output 2d_keypoints_25fps.npy
.\.venv\Scripts\python.exe stage2_3d_lift.py --input 2d_keypoints_25fps.npy --output 3d_points_25fps_frontprior.npy --depth-prior front
.\.venv\Scripts\python.exe stage3_export.py --input 3d_points_25fps_frontprior.npy --bvh motion_25fps_frontprior.bvh --fbx motion_25fps_frontprior.fbx
```

## Quality and coordinate limits

VideoPose3D predicts **pelvis-relative pose**. The default `--root-motion
screen` option approximates image-plane translation from 2D hip movement; it cannot
recover true world position or absolute camera depth. Use `--root-motion fixed`
if the camera pans or zooms. YOLO and the official checkpoint share COCO joint
order, though the checkpoint was trained with Detectron detections, so there is
a detector shift. The rig has no finger or toe tracking because COCO-17 has no
such landmarks. Ordinary clips are grounded using the lower predicted ankle;
image-guided clips also consider the wrists. This can still cause floating or
sinking on difficult clips. Long occlusions,
multiple actors, wide-angle distortion, or truncation can reduce quality.
The frontal depth prior makes the unseen side view more plausible; it cannot
recover the actor's true depth from one fixed camera.

To export a separately named, contact-corrected version of the 25 FPS clip
without overwriting earlier results:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\.venv\Scripts\python.exe stage3_export.py --input 3d_points_25fps_frontprior.npy --bvh motion_25fps_contact.bvh --fbx motion_25fps_contact.fbx --foot-lock on
```

The contact correction moves the whole skeleton by at most 10 cm in the floor
plane and does not change limb proportions or inferred 3D joint angles. For
the current 25 FPS clip, detected contact drift fell from 1.65 cm to 0.65 cm
on average. This measures consistency within inferred contact spans, not
ground-truth foot position.

The VideoPose3D architecture/checkpoint and input normalization follow the
[official code](https://github.com/facebookresearch/VideoPose3D). The pose
extractor uses the [Ultralytics keypoint API](https://docs.ultralytics.com/tasks/pose/),
and the final step uses Blender's [pose bake](https://docs.blender.org/api/current/bpy.ops.nla.html),
[BVH export](https://docs.blender.org/api/5.2/bpy.ops.export_anim.html), and
[FBX export](https://docs.blender.org/api/dev/bpy.ops.export_scene.html) operators.
