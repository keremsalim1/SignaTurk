# SignaTurk / TSL Nexus Runtime

SignaTurk is a FastAPI-based Turkish Sign Language recognition runtime with a browser UI, live webcam inference, prediction history, dictionary views, and a 3D avatar screen. The current runtime integrates the newer SignaTurk 226-class RTMW/RTMPose landmark pipeline on top of the earlier TSL Nexus application shell.

The application is designed to run locally for demos, reports, and development. It can use Supabase/PostgreSQL when configured, but it also starts with a SQLite fallback database when no external database is available.

## What This Project Does

- Opens a webcam stream from the frontend.
- Sends live frames to the FastAPI backend through WebSocket.
- Extracts whole-body and hand landmarks from each frame.
- Converts landmarks into model-ready skeleton, bone, motion, and hand streams.
- Runs an ensemble of SignaTurk Keras models.
- Returns the top predictions with confidence values.
- Stores confident predictions in history.
- Serves dictionary and 3D avatar assets from local landmark data.

## Current Runtime Pipeline

The active pipeline is configured by:

```text
model/signaturk/backend_config.json
```

High-level flow:

```text
RGB webcam frame
-> HF RTMW / RTMPose WholeBody landmark extraction
-> 75 x 4 landmark layout
   pose33 + left_hand21 + right_hand21
-> feature streams
   joint / bone / joint_motion / bone_motion / extra
-> SignaTurk ensemble
   skeleton_v2 + skeleton_v4_seed + skeleton_v6_seed + hand_stream
-> 226-class Turkish Sign Language prediction
```

The default extractor is the local HF RTMW ONNX pipeline when the required ONNX files are present.

## 3D Avatar Animation

The `/avatar3d` screen renders a Mixamo-rigged humanoid and replays sign animations from
the same landmark data used by the dictionary.

High-level flow:

```text
SPA (frontend/app.js) embeds /avatar3d in an iframe
-> word selection sent via postMessage({ type: 'play_sign', word })
-> avatar3d.html fetches GET /landmark/{word}
   each frame: pose33 + left_hand21 + right_hand21 (normalized landmarks)
-> Three.js retargeting onto the avatar skeleton
   position-preserving 2-bone arm IK + wrist/finger aim + torso/head follow
-> rendered with frontend/avatarx.glb
```

Animation quality features in `frontend/avatar3d.html` (all logic is self-contained in this
file; `avatar_solver.js` and `motion_preprocess.js` are no longer loaded):

```text
- Position-preserving 2-bone arm IK (applyArmIK): the wrist target is mapped into avatar space
  with body-anchored anisotropic scaling so signs that bring the hand to the head / face / torso
  or both hands together actually reach those positions (direction-only FK could not).
    * horizontal scale  = avatar shoulder-width / source shoulder-width
    * vertical scale     = avatar (shoulder->face) / source (shoulder->nose)
    * depth (Z) scale    = horizontal * armDepthGain (0.45; MediaPipe pose-z is exaggerated)
- Catmull-Rom cubic interpolation between keyframes + light EMA temporal smoothing.
  NOTE: the data is already offline (Savitzky-Golay) smoothed, so runtime EMA is kept light
  (smoothAlpha 0.85) — heavy runtime smoothing made hands lag behind fast motion.
- Rest-padding trim (trimRestPadding): trims very-low resting frames at clip ends for flow.
- Upper-body focus: waist clipping plane hides the lower body; camera auto-frames the upper body
  (tight zoom that still keeps raised/wide arms on screen), slight 14-degree angle, vertical pan
  so resting hands sit above the control bar.
- Studio gradient background; head/neck follow the nose only weakly (no head dipping).
- Presentation controls: on-screen word caption (top), Replay / Slow (0.6x) / Loop buttons (left).
- delta-time clamped animation loop (stable after tab refocus); ease-out blend back to idle.
```

Tunable parameters live on the `ANIM` object:

```text
useIK=true  armDepthGain=0.45  fingerCurl=16  smoothAlpha=0.85  smoothMaxStep=0.30  FPS=15
wristFullOrient=false  palmRoll=false   (palm-orientation control; off — see limitations below)
```

Known limitations (data/avatar bound, not retarget bugs):

```text
- No hand SHAPE in the data: every sign uses a near-identical open hand (verified; an RTMW
  WholeBody re-extraction pilot produced the same flat hands). Fingers move with the data but
  do not form distinct handshapes. This needs hand-authored handshapes or higher-resolution
  close-up hand capture, not more retarget code.
- Palm roll is uncontrolled by the direction-based wrist aim, so the palm can face the wrong way
  on raised hands. Controlling it from the (flat/noisy) hand-z data jitters during playback, so it
  is disabled. The avatar arm is also short (~60% of expected), so far-reaching signs over-extend.
- Best results come from signs distinguished by arm/wrist motion rather than fine finger shape
  (see DEMO_WORDS.md for a curated, visually verified demo set).
```

The 3D assets (`three.min.js`, `GLTFLoader.js`, `OrbitControls.js`) are served locally
from `frontend/`. The avatar model `frontend/avatarx.glb` is tracked with Git LFS.

### Avatar Landmark Data (generation & repair)

`model/landmarks.json` holds the per-word avatar animation data (226 words, MediaPipe
Holistic layout: pose 9 indices + left_hand 21 + right_hand 21, 30 frames each). It is
generated from the AUTSL Turkish Sign Language video dataset, mapping each word to its
class by name via `SignList_ClassId_TR_EN.csv` + `train_labels.csv`.

The arm/body trajectories are reliable, but the hand landmarks carry no usable finger-shape
detail (the AUTSL clips are too low-resolution for hands). See the "Known limitations" note in
the 3D Avatar Animation section above and `DEMO_WORDS.md` for demo-word guidance.

Two offline tools rebuild/repair this file (need `mediapipe`, `opencv-python`, `scipy`):

```text
tools/extract_landmarks.py   AUTSL videos -> landmarks.json schema (MediaPipe Holistic).
                             Auto-maps word -> AUTSL class -> sample video, picks the best of
                             N candidate samples, merges into the existing file and backs it up.
tools/smooth_landmarks.py    Zero-phase Savitzky-Golay smoothing across the 30 frames to remove
                             jitter without adding lag.
```

Typical regeneration (run from the repo root):

```powershell
# 1) Lock the hand orientation convention against a known-good word
python tools/extract_landmarks.py --validate tesekkur --video <AUTSL>/train/<its_sample>_color.mp4

# 2) Rebuild all words from their correct AUTSL classes
python tools/extract_landmarks.py --all `
  --train-dir <AUTSL>/train `
  --signlist <AUTSL>/SignList_ClassId_TR_EN.csv `
  --train-labels <AUTSL>/train_labels.csv `
  --candidates 2 --model-complexity 2

# 3) De-jitter
python tools/smooth_landmarks.py --window 9 --poly 2
```

The backend caches `landmarks.json` at startup, so restart it after regeneration.

## Repository Layout

```text
backend.py                  FastAPI app, API routes, WebSocket inference, auth/admin endpoints
database.py                 SQLAlchemy engine setup with SQLite fallback
models.py                   SQLAlchemy ORM models for users, history, logs, and settings
landmark_smoother.py        Landmark smoothing for the avatar/dictionary endpoint

frontend/                   React CDN frontend (Babel standalone), styling, 3D avatar screen, Three.js assets
signaturk_runtime/          New SignaTurk runtime: config, extractors, feature builder, ensemble
tools/                      Utility scripts: packaging, runtime checks, and avatar landmark
                            generation/repair (extract_landmarks.py, smooth_landmarks.py)

model/signaturk/            New SignaTurk model config, labels, Keras models, ONNX extractors
model/                      Legacy MediaPipe model files and avatar landmark dictionary
```

## Main Application Files

```text
backend.py
database.py
models.py
landmark_smoother.py
requirements.txt
requirements-gpu.txt
frontend/
signaturk_runtime/
tools/
model/signaturk/
model/landmarks.json
```

## Required Model Assets

The new runtime needs these files:

```text
model/signaturk/backend_config.json
model/signaturk/data/labels/label_map.json
model/signaturk/data/labels/class_display_names.json
model/signaturk/models/skeleton_v2.keras
model/signaturk/models/skeleton_v4_seed.keras
model/signaturk/models/skeleton_v6_seed.keras
model/signaturk/models/hand_stream.keras
model/signaturk/models/rtmw_hf/yolox_m.onnx
model/signaturk/models/rtmw_hf/rtmw-dw-x-l_simcc-cocktail14.onnx
```

The avatar/dictionary section uses:

```text
model/landmarks.json
frontend/avatarx.glb
```

`frontend/avatarx.glb` is the rigged avatar model loaded by the `/avatar3d` screen. It is
tracked with Git LFS (`frontend/*.glb`). Without it the 3D page renders only the floor grid.

## Downloading Model Files

The required model assets are included in this repository through Git LFS. No separate model download is needed when Git LFS is installed correctly.

Recommended clone flow for reviewers:

```powershell
git lfs install
git clone https://github.com/ErayKulkizaga/SignaTurk.git
cd SignaTurk
git lfs pull
```

After cloning, these large files should exist as real files, not tiny text pointer files:

```text
model/model.keras
model/signaturk/models/skeleton_v2.keras
model/signaturk/models/skeleton_v4_seed.keras
model/signaturk/models/skeleton_v6_seed.keras
model/signaturk/models/hand_stream.keras
model/signaturk/models/rtmw_hf/yolox_m.onnx
model/signaturk/models/rtmw_hf/rtmw-dw-x-l_simcc-cocktail14.onnx
frontend/avatarx.glb
```

If GitHub reports an LFS bandwidth/quota error, use the same model files from the provided project bundle and keep the folder paths unchanged.

## Legacy Model Files

The repository also keeps the older 16-frame MediaPipe model family:

```text
model/model.keras
model/demo_config.json
model/norm_stats.json
model/label_map.json
model/label_encoder_classes.npy
```

Do not delete these unless the legacy path is intentionally removed. They are useful for comparison, fallback experiments, and documenting the evolution from the first MediaPipe demo to the newer RTMW ensemble.

## Setup

Requires **Python 3.10–3.12** (the pinned TensorFlow 2.18 / MediaPipe 0.10.21 wheels do
not support Python 3.13+).

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux activate with `source venv/bin/activate` instead.

Install CPU dependencies:

```powershell
pip install -r requirements.txt
```

For GPU-capable environments, use:

```powershell
pip install -r requirements-gpu.txt
```

## Running Locally

Start the backend:

```powershell
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open the app:

```text
http://127.0.0.1:8000
```

Avoid `--reload` during performance checks. TensorFlow, ONNX Runtime, and the landmark extractor are heavy to initialize.

## Environment Variables

`.env` is optional.

If Supabase/PostgreSQL is configured:

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/postgres
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

If `DATABASE_URL` is missing or unavailable, the app falls back to SQLite and may create:

```text
local_dev.db
```

Live inference/debug variables:

```powershell
$env:SIGNATURK_DEBUG_ORIENTATION_VARIANTS="true"
$env:SIGNATURK_LIVE_ORIENTATION="normal"
$env:SIGNATURK_FILTER_MODEL_HANDS="true"
```

Supported orientation modes:

```text
normal
mirror_x
swap_hands
mirror_x_swap
auto
```

`auto` tests multiple orientation variants during prediction and can be slower.

## Useful Endpoints

```text
GET  /                         Frontend app
GET  /api/health               Runtime health check
GET  /api/debug/model-check    Model and config diagnostics
GET  /api/admin/model          Admin model status
GET  /api/history              Prediction history
GET  /signs                    Available avatar/dictionary signs
GET  /landmark/{word}          Smoothed landmark animation data
WS   /api/predict/live         Main live SignaTurk WebSocket
WS   /api/predict/live-legacy  Legacy live WebSocket path
POST /api/predict/sequence     Sequence prediction endpoint
```

## Validation Commands

Syntax check:

```powershell
venv\Scripts\python.exe -m py_compile backend.py signaturk_runtime\config.py signaturk_runtime\realtime_state.py
```

Health check after the server starts:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

Expected healthy state includes:

```text
model_loaded: true
extractor.available: true
extractor.message: HF RTMW wholebody ready
streams: skeleton_v2, skeleton_v4_seed, skeleton_v6_seed, hand_stream
```

## Packaging

To create a clean transferable bundle:

```powershell
.\tools\package_gpu_bundle.ps1
```

The package script excludes development and cache artifacts such as:

```text
.venv/
__pycache__/
.pytest_cache/
dist/
run_logs/
local_dev.db
.cursor/
.vscode/
*.pyc
```

## Troubleshooting

If the app starts but Supabase fails, this is acceptable for local use as long as SQLite fallback is active.

If predictions are unstable or low-confidence:

- Check that the HF RTMW ONNX files exist under `model/signaturk/models/rtmw_hf/`.
- Confirm `/api/health` reports `HF RTMW wholebody ready`.
- Test `SIGNATURK_LIVE_ORIENTATION` values if camera mirroring or left/right hand orientation appears wrong.
- Keep `SIGNATURK_FILTER_MODEL_HANDS=true` unless intentionally debugging raw landmarks.
- Avoid running with `--reload` while measuring inference latency.

If the frontend opens but the camera does not start:

- Use `http://127.0.0.1:8000`, not a random file path.
- Allow camera permission in the browser.
- Make sure another browser tab or app is not already locking the webcam.

## External Reference Folders

These folders are not part of this repository, but may be useful during development:

```text
C:\Users\Eray\Documents\SignaTurk
```

Training/notebook and demo backend source for the newer 226-class RTMPose/RTMW model, if present locally.

```text
C:\Users\Eray\SEZAR\PROJECTS\SignLanguage\demo
```

Older 16-frame MediaPipe demo used for comparison with the current runtime. Treat it as a read-only reference unless intentionally migrating code.
