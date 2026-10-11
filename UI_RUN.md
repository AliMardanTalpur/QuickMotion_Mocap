# QuickMotion local UI

The UI implements the four Figma frames: Home, Assets, Single Camera upload,
and Motion Capture Asset. Share and Docs in the sidebar also work. It uses the
existing Python environment and a local browser; no Node build or cloud
account is needed.

## Start

Open PowerShell in the project folder:

```powershell
cd 'C:\Users\Ali Mardan\Documents\ChatGPT\MOCAP'
.\.venv\Scripts\python.exe ui_server.py --open
```

If the browser does not open automatically, visit <http://127.0.0.1:8765/>.
Keep the PowerShell window open while using QuickMotion. Press **Ctrl+C** to
stop the server. To use a different local port, add `--port 8766`.
Starting the same command again while the UI is open now opens the existing
server without changing an active job. Keep the original PowerShell window
open until processing finishes.

The existing `dance.mp4` result appears in Recent assets as a sample. You can
inspect its source video, rotate and play the 3D skeleton, view the alignment
sheet, and download its FBX/BVH or a ZIP share package immediately.

## Process a new video

1. Click **Upload** or **Single Cam**, then choose/drop an MP4, AVI, or MOV up
   to 1080p and 60 FPS. The limit is 2 GB per upload.
2. Leave **Higher quality mirrored pass** enabled for the best GVHMR result.
3. Open the new asset to watch its stage, progress, and log. Queued videos run
   one at a time. The GPU is not shared by models; GVHMR uses CPU on this
   machine to fit its RTX 2050's 4 GB VRAM. Full HD clips can take tens of
   minutes. The animated bar during 3D recovery indicates work is underway;
   its starting percentage is not a per-frame estimate.
4. Once complete, play and rotate the 3D preview, then download FBX/BVH. Use
   **Download share package** to get the source, animations, preview, and
   quality report in one ZIP.

The UI launches `gvhmr_infer.py`, `gvhmr_refine.py`, `motion_preview.py`,
`gvhmr_review.py`, and `stage3_export.py` sequentially. Results and persistent
job manifests are stored under `runs\quickmotion\<asset-id>\`. Failed or
interrupted jobs can be retried from their detail page. New jobs need the
registered SMPL-X model and four GVHMR checkpoints; the upload screen reports
missing files. See [GVHMR_RUN.md](GVHMR_RUN.md) for setup.

**Share** packages outputs for transfer. A copied local link opens only while
this server is running on this computer. QuickMotion binds to `127.0.0.1`.

The browser's 3D preview is a joint visualization. Import the FBX in Blender
for the full animated armature. Single-camera depth and hidden limbs remain
estimates.
