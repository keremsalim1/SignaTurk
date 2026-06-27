# Project Status

Last updated: 2026-06-25

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

## Frontend / 3D Avatar Runtime

3D animasyon ekrani:

```text
frontend/avatar3d.html      Three.js sahne, GLB yukleme, landmark -> kemik retargeting (TUM mantik burada)
frontend/avatarx.glb        Mixamo-rigli humanoid avatar (Git LFS)
frontend/three.min.js, GLTFLoader.js, OrbitControls.js   Lokal Three.js r128 kopyalari
frontend/avatar_solver.js, motion_preprocess.js   ARTIK YUKLENMIYOR (script etiketleri kaldirildi; olu kod)
```

SPA (`frontend/app.js`) bu sayfayi bir iframe icinde gosterir ve secilen kelimeyi
`postMessage({type:'play_sign', word})` ile gonderir. Avatar3d sayfasi `/landmark/{word}`
endpoint'inden kare dizisini ceker (her kare: pose33 + left_hand21 + right_hand21
normalize landmark) ve avatar iskeletine uygular.

Son frontend duzeltmeleri:

```text
index.html: @babel/standalone classic JSX runtime'a zorlandi
            (otomatik runtime "import" uretip iframe disinda hata veriyordu).
iframe src: surum sorgusu (?v=...) ile cache-bust eklendi.
avatar3d.html: avatar yukleme blogu sadelestirildi, gereksiz clipping plane kaldirildi,
            model 1.8 birime olceklenip zemine oturtuluyor.
Koordinat donusumu: lmVec/handOffsetVec X/Z ekseni referansa geri alindi
            (ayna/sirt-donuk hareket duzeldi).
Hareket kalitesi: Catmull-Rom kubik interpolasyon, EMA temporal yumusatma +
            landmark hiz kisiti, kelimeler arasi smoothstep gecis, kol erisimi artirildi.
```

## Avatar Veri Yeniden Uretimi + Animasyon Duzeltmeleri (2026-06-25)

Avatar landmark verisi (`model/landmarks.json`) AUTSL veri setinden bastan uretildi.

Kok neden: deployed dosyada "iyi" sanilan 131 kelimenin 111'i YANLIS sinif videosundan
etiketlenmisti (sinif-eslemesi kaymasi; or. deployed `tesekkur` aslinda `pazartesi`
isaretiydi). Geri kalan 96 kelime ise dejenere/placeholder veriydi (~%19 near-duplicate).

Cozum: kelime ADIYLA SignList'ten dogru AUTSL sinifina eslenip videodan yeniden cikarildi.
Bu yalnizca avatar (text->isaret) verisini etkiler; recognition modeli (class_id tabanli)
degismez.

Yeni offline araclar:

```text
tools/extract_landmarks.py   AUTSL videosu -> landmarks.json (MediaPipe Holistic, 30 kare).
                             Kelime -> SignList sinifi -> ornek video otomatik eslenir;
                             en iyi N adaydan secilir; mevcut dosyaya merge + .bak.
tools/smooth_landmarks.py    Faz-kaydirmasiz Savitzky-Golay yumusatma (titreme giderme).
```

Dogrulama (yeniden uretim sonrasi): yanlis-esleme 0, kopya 0, hepsi source_video'lu,
medyan kalite ~98.7; titreme poz 0.0138->0.0054 / el 0.0058->0.0032.
(Not: sonradan artik `class_191` anahtari silindi -> kelime sayisi 227 -> **226**, AUTSL sinif
sayisiyla birebir. Yedek: `model/landmarks.json.bak_class191`.)

`frontend/avatar3d.html` animasyon duzeltmeleri:

```text
- Bel alti animasyonu kapatildi (Hips/Spine sabit & dik), bel hizasinda clipping plane ile
  alt govde gizlendi, kamera ust govdeye cerceveledi.
- Eller her zaman gorunur: kesim cizgisi idle ellerin biraz altina adaptif konuldu.
- HAND_DEPTH_GAIN=1.2 (MediaPipe el-z sikismasini telafi; 2.2 parmaklari capraziyordu).
```

Yeni bagimlilik: `scipy` (offline yumusatma araci icin requirements'a eklendi).

Not: backend `landmarks.json`'u baslangicta cache'ler -> yeniden uretimden sonra restart gerekir.

## Animasyon Retarget Revizyonu + Tavan Analizi (2026-06-25, devam)

Tum degisiklikler `frontend/avatar3d.html` icinde, genel kod (kelimeye bagli degil, 226 kelimede gecerli).

### Kol/govde retarget (KONUM artik dogru)
- **Konum-koruyan 2-kemik IK** (`applyArmIK`) eski yon-tabanli FK'nin yerini aldi. Yon-tabanli FK
  yalnizca aciyi taklit edip konumu atiyordu -> el kafaya/govdeye/karsi-ele YAKLASAMIYORDU.
- **Govde-capali anizotropik olcek** (`ikMapPoint`): bilek hedefi avatar uzayina tasinirken
  - yatay (X) olcek = avatar omuz-genisligi / kaynak omuz-genisligi,
  - dikey (Y) olcek = avatar (omuz->YUZ/burun hizasi) / kaynak (omuz->burun)  [Head kemigi kafa
    tabaninda; once kafa-TEPESI denendi ama el-opme/bayram'da eli kafanin ustune tasiyordu ->
    yuz hizasina indirildi],
  - derinlik (Z) olcek = X * `armDepthGain` (varsayilan 0.45; MediaPipe pose-z'si abartili oldugundan
    1.0'de eller govdeden one firliyordu).
- Kol kemik uzunluklari rest pozda olculur (`ARM_LEN`). Avatarx.glb kolu KISA (omuz->bilek 0.477;
  1.8m boy icin ~%60) -> cok uzanan isaretlerde kol tam acilip gerginlesir (avatar rig sinirli).

### Yumusatma (en onemli akis duzeltmesi)
- Veri zaten offline Savitzky-Golay yumusak; runtime EMA `smoothAlpha=0.55` USTUNE binince
  cift-yumusatma yapip elleri HIZLI harekette hedefin GERISINDE birakiyordu (lag).
  -> `smoothAlpha=0.85`, `smoothMaxStep=0.30`. "Eller yanlis yere gidiyor"un ana sebebi buydu.
- `trimRestPadding`: klip bas/son cok-alcak (dinlenme) karelerini buda; akis duzelir. FPS=15.

### Parmaklar + avuc yonu (VERI SINIRI — kod degil)
- **Veride el-SEKLI yok** (kanit: 2B parmak kivrim orani ~0.96-1.0 TUM kelimeler; parmak ucu-bilek
  mesafeleri kelimeler arasi sabit). Parmaklar landmark'tan SURULUR (kare-kare hareket eder, donuk
  degil) + hafif kozmetik kivrim, ama belirgin handshape YOK.
- **RTMW WholeBody pilotu** (rtmlib, 5 kelime) bunu CURUTTU: RTMW de parmaklari duz/acik veriyor,
  hatta bazi kelimede mevcut veriden daha duz. Yani AUTSL videolari (dusuk coz./kucuk el/bulaniklik)
  el-sekli tasimyor -> 226 yeniden cikarim BOSA gider. Tam-cikarim YAPMA.
- **Avuc yonu (roll):** yon-tabanli bilek aim roll'u sabitlemiyor -> yukari hareketlerde avuc bazen
  ters bakiyor. Kontrol denendi (full-orient `wristFullOrient`, cerrahi `palmRoll`) -> avuc dogru
  oluyor AMA oynatimda titriyor (duz/gurultulu el-z). Ikisi de KAPALI/geri alindi. Veri sinirli.

### Sunum / UI
- Caption viewport USTUNE (`top`); kontrol cubugu (Tekrar/Yavas/Dongu) SOL-alta dikey (eli engellemesin).
- Kamera: dikey+yatay acikliktan hesaplanan sikistirma (kalkik+genis eller sigar) + hafif 14 derece
  yan aci (derinlik) + `bottomPan` (alt eller kontrol cubugunun ustunde kalir).
- Arka plan duz siyahtan `makeStudioBg()` slate dikey gradyana (siyah sac kayboluyordu).
- Kafa/boyun burun takibi zayiflatildi (0.40/0.35 -> 0.14/0.12) -> harekette bas asagi egilmiyor.
- Yavas mod 0.6 (eskiden 0.4).

### Aktif tunable bayraklar (ANIM uzerinde)
```text
useIK=true  armDepthGain=0.45  fingerCurl=16  smoothAlpha=0.85  smoothMaxStep=0.30  FPS=15
wristFullOrient=false  palmRoll=false  forearmTwist (varsayilan acik, etkisi olcumde ~0)
```
Iframe surumu (`frontend/app.js` icindeki `/avatar3d?v=...`) her UI degisikliginde bumplaniyor
(cache-bust); en son `ui7`.

### TAVAN ANALIZI (bu veri+avatar icin maksimuma yakin)
- Kol/govde KONUMU ~optimal; daha fazla retarget kodu tavani yukseltmez (palmRoll/full-orient/
  finger-override denemeleri ya hareketi bozdu ya verisiz ise yaramadi).
- Tavani yukselten 2 sey KODDA DEGIL:
  1. **El-sekli** (veri sinirli) -> elle authoring (~2 hafta/226, TID bilgisi gerekir) veya
     yuksek-coz. yakin-el yeniden cekim.
  2. **Yuz/mimik non-manual** (avatar sinirli) -> yuz blendshape'li, uzun-kollu daha iyi avatar.
- Orta vadeli kod kazanci: co-articulation (kelimeler arasi nötre donmeden blend), self-collision.

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
frontend/avatarx.glb
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
