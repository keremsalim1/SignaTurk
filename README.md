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

## Repository Layout

```text
backend.py                  FastAPI app, API routes, WebSocket inference, auth/admin endpoints
database.py                 SQLAlchemy engine setup with SQLite fallback
models.py                   SQLAlchemy ORM models for users, history, logs, and settings
landmark_smoother.py        Landmark smoothing for the avatar/dictionary endpoint

frontend/                   React CDN frontend, styling, avatar screen, static assets
signaturk_runtime/          New SignaTurk runtime: config, extractors, feature builder, ensemble
tools/                      Utility scripts for packaging and runtime checks

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
```

Large model files are stored with Git LFS. Install Git LFS before cloning if you need the repository to download the Keras, ONNX, NumPy, and MediaPipe task assets automatically:

```powershell
git lfs install
git clone https://github.com/ErayKulkizaga/SignaTurk.git
```

If the repository was cloned before Git LFS was installed, run:

```powershell
git lfs pull
```

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

Create or activate a Python virtual environment:

```powershell
.venv\Scripts\activate
```

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
.venv\Scripts\python.exe -m py_compile backend.py signaturk_runtime\config.py signaturk_runtime\realtime_state.py
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
