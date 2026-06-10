# TSL Nexus / SignaTurk Runtime

FastAPI + React CDN uygulamasi. Webcam goruntusunden Turk Isaret Dili tahmini yapar, tahmin gecmisini tutar ve sozluk/3D avatar ekranlarini sunar.

## Aktif Mimari

```text
backend.py                  FastAPI uygulamasi, auth, admin API, live inference
database.py                 SQLAlchemy engine ve SQLite fallback
models.py                   User, History, Log, Setting ORM modelleri
landmark_smoother.py        /landmark/{word} animasyon verisi smoothing
frontend/                   React CDN arayuzu ve 3D avatar
signaturk_runtime/          Yeni SignaTurk 226-class RTMPose runtime
model/signaturk/            Yeni model config, labels, Keras/ONNX modelleri
model/                      Legacy 16-frame MediaPipe model ve landmarks.json
tools/                      Calistirma, model indirme ve paketleme scriptleri
```

Yeni ana pipeline `model/signaturk/backend_config.json` ile yuklenir:

```text
RGB frame -> RTMPose WholeBody -> 75x4 landmarks
-> joint / bone / motion / extra streams
-> skeleton_v2 + skeleton_v4_seed + skeleton_v6_seed + hand_stream ensemble
```

Legacy pipeline hala kodda durur ve `model/model.keras`, `model/demo_config.json`, `model/norm_stats.json`, `model/label_map.json` dosyalarini kullanir. Bu dosyalari silme; yeni runtime yuklenemezse veya karsilastirma gerekiyorsa lazim olabilir.

## Calistirma

Windows:

```powershell
.venv\Scripts\activate
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Tarayici:

```text
http://127.0.0.1:8000
```

Reload performans testi icin onerilmez; TensorFlow ve RTMPose modelleri agir yuklenir.

## Ortam Degiskenleri

`.env` opsiyoneldir. Supabase kullanilacaksa:

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/postgres
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

`DATABASE_URL` yoksa proje SQLite fallback ile `local_dev.db` olusturabilir.

Canli inference debug ayarlari:

```powershell
$env:SIGNATURK_DEBUG_ORIENTATION_VARIANTS="true"
$env:SIGNATURK_LIVE_ORIENTATION="normal"        # normal | mirror_x | swap_hands | mirror_x_swap | auto
$env:SIGNATURK_FILTER_MODEL_HANDS="true"
```

`auto` her tahminde tum orientation varyantlarini kostugu icin yavas olabilir.

## Gerekli Model Dosyalari

Yeni runtime icin minimum:

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

Avatar sozlugu icin:

```text
model/landmarks.json
```

## Temiz Paket Olusturma

```powershell
.\tools\package_gpu_bundle.ps1
```

Paketleme `.venv`, cache, `dist`, editor ayarlari ve yerel veritabani dosyalarini dahil etmez.

## Kontrol Komutlari

```powershell
.venv\Scripts\python.exe -m py_compile backend.py signaturk_runtime\config.py signaturk_runtime\realtime_state.py
```

Saglik endpointleri:

```text
/api/health
/api/debug/model-check
/api/admin/model
```

## Dis Referans Klasorleri

Bu repo disinda iki faydali referans var:

```text
C:\Users\Eray\Documents\SignaTurk
```

Yeni 226-class RTMPose modelinin Colab/notebook ve demo backend kaynagi.

```text
C:\Users\Eray\SEZAR\PROJECTS\SignLanguage\demo
```

Eski 16-frame MediaPipe demo. Canli landmark akisi ve legacy model davranisini karsilastirmak icin okunabilir; bu klasor bu runtime'in parcasi degildir.
