# Project Status

Last updated: 2026-06-10

## Current Goal

TSL Nexus uygulamasina yeni SignaTurk RTMPose tabanli 226-class modeli entegre ediliyor.

Eski TSL Nexus modeli 16-frame MediaPipe tabanliydi. Yeni runtime ise 32-frame RTMPose WholeBody skeleton stream ensemble kullaniyor.

## Active Runtime

Aktif backend:

```text
backend.py
```

Aktif frontend:

```text
frontend/index.html
frontend/app.js
frontend/style.css
frontend/avatar3d.html
```

Aktif yeni runtime:

```text
signaturk_runtime/
model/signaturk/
```

Legacy model dosyalari da korunuyor:

```text
model/model.keras
model/demo_config.json
model/label_map.json
model/norm_stats.json
model/label_encoder_classes.npy
```

## Live Pipeline Notes

Yeni live endpoint:

```text
/api/predict/live
```

Akis:

```text
browser JPEG frame
-> backend decode
-> RTMPose extractor
-> 75x4 landmarks
-> FrameBuffer, 32 frame target
-> SignaTurk ensemble
-> top predictions + diagnostics
```

Mevcut debug alanlari:

```text
debug.frame
debug.buffer
debug.preprocess
debug.timing
debug.variants
```

Orientation test ayarlari:

```text
SIGNATURK_LIVE_ORIENTATION=normal|mirror_x|swap_hands|mirror_x_swap|auto
SIGNATURK_DEBUG_ORIENTATION_VARIANTS=true
```

Son backend degisiklikleri:

```text
RTMPose inference asyncio.to_thread ile event loop disina alindi.
Model girdisine gitmeden once el landmark filtresi eklendi.
75x4 layout icin orientation transformlari eklendi.
Frontend diagnostics orientation_applied gosteriyor.
```

## Known Investigation Area

Canli confidence dusukse ana supheli alanlar:

```text
1. Webcam/RTMPose landmark kalitesi
2. Mirror veya hand slot uyumsuzlugu
3. 32-frame live segmentin egitim segmentinden farkli temporal dagilimi
4. RTMPose ile eski MediaPipe pipeline arasindaki feature dagilimi farki
```

Eski 16-frame demo daha basit calisiyordu:

```text
browser MediaPipe hands -> 16 frame buffer -> z-score -> BiLSTM
```

Yeni pipeline daha guclu ama canli tarafta egitim pipeline'ina daha hassas.

## Keep / Do Not Delete

Korunacaklar:

```text
backend.py
database.py
models.py
landmark_smoother.py
frontend/
signaturk_runtime/
tools/
model/signaturk/
model/landmarks.json
model/model.keras and legacy model metadata
requirements.txt
requirements-gpu.txt
README.md
PROJECT_STATUS.md
```

Temizlenebilir uretilmis dosyalar:

```text
__pycache__/
.pytest_cache/
dist/
run_logs/
local_dev.db
*.pyc
.cursor/
.vscode/
```

`.venv/` kaynak kod degildir ama lokal calistirma icin tutulabilir.
