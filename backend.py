"""
TSL Nexus — Backend v3.0
Endpoints:
  /                        → frontend/index.html (React SPA)
  /avatar3d                → frontend/avatar3d.html (Three.js avatar)
  /api/auth/*              → login, register
  /api/predict/live        → WebSocket — real-time sign prediction
  /api/predict/sequence    → POST — one-shot prediction
  /api/history             → prediction history
  /api/settings            → app settings
  /api/dictionary          → 226-class label lookup
  /api/admin/*             → overview, users, logs, model stats
  /api/debug/model-check   → full diagnostics
  /signs                   → list 3D landmark words
  /landmark/{word}         → landmark JSON for 3D avatar
"""

import os, json, time, traceback, uuid, logging, base64, asyncio
from contextlib import asynccontextmanager
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

import database as db_module
from database import engine, get_db, Base
import models
from signaturk_runtime.config import default_settings as signaturk_default_settings
from signaturk_runtime.config import load_backend_config as signaturk_load_backend_config
from signaturk_runtime.config import load_display_names as signaturk_load_display_names
from signaturk_runtime.config import load_id2label as signaturk_load_id2label
from signaturk_runtime.feature_builder import (
    HAND_COUNT as ST_HAND_COUNT,
    LEFT_HAND_NODES as ST_LEFT_HAND_NODES,
    POSE_COUNT as ST_POSE_COUNT,
    RIGHT_HAND_NODES as ST_RIGHT_HAND_NODES,
    TOTAL_LANDMARKS as ST_TOTAL_LANDMARKS,
    build_feature_bundle as signaturk_build_feature_bundle,
    sample_indices as signaturk_sample_indices,
)
from signaturk_runtime.model_loader import SignaTurkEnsemble
from signaturk_runtime.rtmpose_extractor import RTMPoseExtractor

# ── Landmark smoother (senin modülün) ────────────────────
try:
    from landmark_smoother import smooth_landmark_data
    SMOOTHER_AVAILABLE = True
except ImportError:
    SMOOTHER_AVAILABLE = False
    print("[UYARI] landmark_smoother modülü bulunamadı — /landmark endpoint ham veri döndürür")

# ── TensorFlow ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    tf.get_logger().setLevel("ERROR")
except AttributeError:
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("tsl")

# ═══════════════════════════════════════════════════════════
#  PASSWORD HASHING
# ═══════════════════════════════════════════════════════════
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def hash_pw(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_pw(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR    = BASE_DIR / "model"

# ═══════════════════════════════════════════════════════════
#  GLOBALS — Model (226 sınıf)
# ═══════════════════════════════════════════════════════════
MODEL        = None
LABEL_MAP    = {}
NORM_MEAN    = None
NORM_STD     = None
DEMO_CONFIG  = {}
MP_HANDS     = None
SIGNATURK_SETTINGS = None
SIGNATURK_CONFIG = {}
SIGNATURK_ENSEMBLE = None
SIGNATURK_EXTRACTOR = None
SIGNATURK_ID2LABEL = {}
SIGNATURK_DISPLAY_NAMES = {}
SIGNATURK_REALTIME_STREAMS = []

SEQ_LEN      = 32
FEAT_DIM     = 300
NUM_CLASSES  = 226
CONFIDENCE_THRESHOLD = 0.70
MODEL_LOAD_TIME  = 0.0
AVG_INFERENCE_MS = 0.0
INFERENCE_COUNT  = 0
LIVE_DEBUG_ENABLED = True
LIVE_VARIANTS_ENABLED = False
ACTIVE_DB_LABEL = "Supabase PostgreSQL"

# Landmark verisi (3D animasyon için) — tek birleşik dosya
LANDMARKS_FILE = MODEL_DIR / "landmarks.json"
LANDMARK_INDEX = {}  # {word: data_dict}

# ═══════════════════════════════════════════════════════════
#  MEDIAPIPE INIT
# ═══════════════════════════════════════════════════════════
def init_mediapipe():
    for strategy_num, init_fn in enumerate([
        lambda: __import__('mediapipe').solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.3, min_tracking_confidence=0.3),
        lambda: __import__('mediapipe.python.solutions.hands', fromlist=['Hands']).Hands(
            static_image_mode=True, max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.3, min_tracking_confidence=0.3),
    ], start=1):
        try:
            h = init_fn()
            h.process(np.zeros((100, 100, 3), dtype=np.uint8))
            logger.info(f"[MEDIAPIPE] Strategy {strategy_num} OK")
            return h
        except Exception as e:
            logger.warning(f"[MEDIAPIPE] Strategy {strategy_num} failed: {e}")
    logger.error("[MEDIAPIPE] All strategies failed")
    return None


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════
def legacy_load_model_assets():
    """Arkadaşının tsl-nexus modelini yükle."""
    global MODEL, LABEL_MAP, NORM_MEAN, NORM_STD, DEMO_CONFIG, MP_HANDS
    global SEQ_LEN, FEAT_DIM, NUM_CLASSES, CONFIDENCE_THRESHOLD, MODEL_LOAD_TIME

    t0 = time.time()

    cfg_path = MODEL_DIR / "demo_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            DEMO_CONFIG = json.load(f)
        SEQ_LEN              = DEMO_CONFIG.get("seq_len", 16)
        FEAT_DIM             = DEMO_CONFIG.get("feat_dim", 156)
        NUM_CLASSES          = DEMO_CONFIG.get("num_classes", 179)
        CONFIDENCE_THRESHOLD = DEMO_CONFIG.get("confidence_threshold", 0.40)
        logger.info(f"[CONFIG] seq_len={SEQ_LEN} feat_dim={FEAT_DIM} num_classes={NUM_CLASSES}")

    lm_path = MODEL_DIR / "label_map.json"
    if lm_path.exists():
        with open(lm_path, encoding="utf-8") as f:
            LABEL_MAP = json.load(f)
        logger.info(f"[LABELS] {len(LABEL_MAP)} sınıf yüklendi")

    ns_path = MODEL_DIR / "norm_stats.json"
    if ns_path.exists():
        with open(ns_path) as f:
            ns = json.load(f)
        NORM_MEAN = np.array(ns["mean"], dtype=np.float32)
        NORM_STD  = np.where(np.array(ns["std"], dtype=np.float32) < 1e-6, 1.0,
                             np.array(ns["std"], dtype=np.float32))

    model_path = MODEL_DIR / "model.keras"
    if model_path.exists():
        try:
            @tf.keras.utils.register_keras_serializable()
            class ReduceSumAxis1(tf.keras.layers.Layer):
                def call(self, inputs):
                    return tf.reduce_sum(inputs, axis=1)
                def compute_output_shape(self, input_shape):
                    return (input_shape[0], input_shape[2])

            MODEL = tf.keras.models.load_model(str(model_path))
            MODEL.predict(np.zeros((1, SEQ_LEN, FEAT_DIM), dtype=np.float32), verbose=0)
            logger.info(f"[MODEL-NEXUS] Yüklendi — {MODEL.input_shape} → {MODEL.output_shape}")
        except Exception as e:
            logger.error(f"[MODEL-NEXUS] Yüklenemedi: {e}")

    MP_HANDS = init_mediapipe()
    MODEL_LOAD_TIME = time.time() - t0
    logger.info(f"[STARTUP] tsl-nexus assets {MODEL_LOAD_TIME:.2f}s'de yüklendi")


def load_model_assets():
    """Load the SignaTurk 226-class ensemble and RGB-to-skeleton extractor."""
    global MODEL, LABEL_MAP, NORM_MEAN, NORM_STD, DEMO_CONFIG, MP_HANDS
    global SIGNATURK_SETTINGS, SIGNATURK_CONFIG, SIGNATURK_ENSEMBLE, SIGNATURK_EXTRACTOR
    global SIGNATURK_ID2LABEL, SIGNATURK_DISPLAY_NAMES, SIGNATURK_REALTIME_STREAMS
    global SEQ_LEN, FEAT_DIM, NUM_CLASSES, CONFIDENCE_THRESHOLD, MODEL_LOAD_TIME

    t0 = time.time()
    NORM_MEAN = None
    NORM_STD = None

    try:
        SIGNATURK_SETTINGS = signaturk_default_settings()
        SIGNATURK_CONFIG = signaturk_load_backend_config(SIGNATURK_SETTINGS)
        SIGNATURK_ID2LABEL = signaturk_load_id2label(SIGNATURK_SETTINGS)
        SIGNATURK_DISPLAY_NAMES = signaturk_load_display_names(SIGNATURK_SETTINGS)
        runtime_cfg = SIGNATURK_CONFIG.get("runtime", {})
        streams_env = os.getenv("SIGNATURK_REALTIME_STREAMS", "").strip()
        if streams_env:
            SIGNATURK_REALTIME_STREAMS = [item.strip() for item in streams_env.split(",") if item.strip()]
        else:
            SIGNATURK_REALTIME_STREAMS = list(runtime_cfg.get("realtime_streams") or SIGNATURK_CONFIG.get("streams", []))

        SEQ_LEN = int(SIGNATURK_CONFIG.get("frame_count", 32))
        NUM_CLASSES = int(SIGNATURK_CONFIG.get("num_classes", 226))
        FEAT_DIM = int(SIGNATURK_CONFIG.get("feature_shapes", {}).get("joint", [SEQ_LEN, 300])[-1])
        CONFIDENCE_THRESHOLD = float(getattr(SIGNATURK_SETTINGS, "top1_threshold", 0.70))
        DEMO_CONFIG = {
            **SIGNATURK_CONFIG,
            "seq_len": SEQ_LEN,
            "feat_dim": "ensemble: joint/bone/joint_motion/bone_motion/extra",
            "preprocessing": "RGB frames -> RTMPose WholeBody -> 75x4 landmarks -> skeleton streams",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }
        LABEL_MAP = {
            str(model_idx): {
                "TR": SIGNATURK_DISPLAY_NAMES.get(raw_label, f"class_{raw_label}"),
                "EN": f"class_{raw_label}",
            }
            for model_idx, raw_label in SIGNATURK_ID2LABEL.items()
        }
        logger.info(f"[SIGNATURK-CONFIG] frames={SEQ_LEN} classes={NUM_CLASSES} streams={SIGNATURK_REALTIME_STREAMS}")
        logger.info(f"[SIGNATURK-LABELS] {len(LABEL_MAP)} classes loaded")

        SIGNATURK_ENSEMBLE = SignaTurkEnsemble(SIGNATURK_SETTINGS)
        config_errors = SIGNATURK_ENSEMBLE.validate_config()
        if config_errors:
            raise RuntimeError("; ".join(config_errors))
        SIGNATURK_ENSEMBLE.load()
        MODEL = SIGNATURK_ENSEMBLE
        logger.info(f"[SIGNATURK-MODEL] Ensemble loaded: {list(SIGNATURK_ENSEMBLE.models)}")
        if os.getenv("SIGNATURK_MODEL_WARMUP", "true").lower() in {"1", "true", "yes"}:
            warmup_ms = SIGNATURK_ENSEMBLE.warmup(streams=SIGNATURK_REALTIME_STREAMS or None)
            logger.info(f"[SIGNATURK-MODEL] Warm-up completed in {warmup_ms:.1f}ms")
    except Exception as e:
        MODEL = None
        SIGNATURK_ENSEMBLE = None
        logger.error(f"[SIGNATURK-MODEL] Could not load: {e}")

    try:
        runtime_cfg = SIGNATURK_CONFIG.get("runtime", {})
        SIGNATURK_EXTRACTOR = RTMPoseExtractor(
            preferred_extractor=runtime_cfg.get("preferred_extractor", "rtmpose"),
            try_rtmpose_first=runtime_cfg.get("try_rtmpose_first", True),
            allow_mediapipe_fallback=runtime_cfg.get("allow_mediapipe_fallback", False),
        )
        MP_HANDS = SIGNATURK_EXTRACTOR if SIGNATURK_EXTRACTOR.status.available else None
        logger.info(f"[SIGNATURK-EXTRACTOR] {SIGNATURK_EXTRACTOR.status.message}")
    except Exception as e:
        SIGNATURK_EXTRACTOR = None
        MP_HANDS = None
        logger.error(f"[SIGNATURK-EXTRACTOR] Could not start: {e}")

    MODEL_LOAD_TIME = time.time() - t0
    logger.info(f"[STARTUP] SignaTurk assets loaded in {MODEL_LOAD_TIME:.2f}s")


def load_landmark_index():
    """Birleşik landmark.json dosyasını belleğe yükle (3D avatar için)."""
    global LANDMARK_INDEX

    if LANDMARKS_FILE.exists():
        with open(LANDMARKS_FILE, "r", encoding="utf-8") as f:
            LANDMARK_INDEX = json.load(f)
        logger.info(f"[LANDMARKS] {len(LANDMARK_INDEX)} kelime yüklendi")
    else:
        logger.warning(f"[LANDMARKS] {LANDMARKS_FILE} bulunamadı")


# ═══════════════════════════════════════════════════════════
#  PREPROCESSING — tsl-nexus
# ═══════════════════════════════════════════════════════════
def extract_landmarks_from_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if MP_HANDS is None:
        return None
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = MP_HANDS.process(rgb)
        left_lm = np.zeros(63, dtype=np.float32)
        right_lm = np.zeros(63, dtype=np.float32)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand.classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark],
                                   dtype=np.float32).flatten()
                if label == "Left":
                    left_lm = coords
                else:
                    right_lm = coords
        return np.concatenate([left_lm, right_lm])
    except Exception as e:
        logger.error(f"[MEDIAPIPE] Frame error: {e}")
        return np.zeros(126, dtype=np.float32)


def normalize_hands_relative(X):
    result = X.copy()
    T = result.shape[0]
    for start in [0, 63]:
        hand = result[:, start:start+63].reshape(T, 21, 3)
        hand_rel = hand - hand[:, 0:1, :]
        scale = np.linalg.norm(hand_rel[:, 9, :], axis=-1, keepdims=True)[:, :, np.newaxis]
        scale = np.where(scale < 1e-6, 1.0, scale)
        result[:, start:start+63] = (hand_rel / scale).reshape(T, 63)
    return result


def compute_finger_angles(X):
    T = X.shape[0]
    chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    angles_all = []
    for hs in [0, 63]:
        hand = X[:, hs:hs+63].reshape(T, 21, 3)
        ha = np.zeros((T, 15), dtype=np.float32)
        idx = 0
        for chain in chains:
            for i in range(len(chain) - 1):
                a = hand[:, (0 if i == 0 else chain[i-1]), :]
                b = hand[:, chain[i], :]
                c = hand[:, chain[i+1], :]
                v1, v2 = a - b, c - b
                cos_a = np.sum(v1*v2, axis=-1) / (
                    np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
                ha[:, idx] = np.arccos(np.clip(cos_a, -1, 1))
                idx += 1
        angles_all.append(ha)
    return np.concatenate([X, np.concatenate(angles_all, axis=-1)], axis=-1)


def zscore_normalize(X):
    if NORM_MEAN is None or NORM_STD is None:
        return X
    return (X - NORM_MEAN) / NORM_STD


def preprocess_sequence(raw_landmarks):
    x = normalize_hands_relative(raw_landmarks)
    x = compute_finger_angles(x)
    x = zscore_normalize(x)
    return x[np.newaxis, :, :].astype(np.float32)


def _safe_float(value, digits=4):
    return round(float(value), digits)


def _hand_debug_stats(hand_values):
    hand = np.asarray(hand_values, dtype=np.float32).reshape(21, 3)
    present = bool(np.max(np.abs(hand)) > 1e-6)
    stats = {
        "present": present,
        "nonzero_ratio": _safe_float(np.count_nonzero(np.abs(hand) > 1e-6) / hand.size),
    }
    if not present:
        stats.update({"bbox": None, "range": None})
        return stats

    stats["bbox"] = {
        "x": [_safe_float(hand[:, 0].min()), _safe_float(hand[:, 0].max())],
        "y": [_safe_float(hand[:, 1].min()), _safe_float(hand[:, 1].max())],
        "z": [_safe_float(hand[:, 2].min()), _safe_float(hand[:, 2].max())],
    }
    stats["range"] = [_safe_float(hand.min()), _safe_float(hand.max())]
    return stats


def summarize_landmarks(landmarks):
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1)
    if arr.shape != (126,):
        return {"shape": list(arr.shape), "valid": False}

    left = _hand_debug_stats(arr[:63])
    right = _hand_debug_stats(arr[63:])
    hands_detected = int(left["present"]) + int(right["present"])
    return {
        "shape": [126],
        "valid": True,
        "layout": "left_hand(63)+right_hand(63)",
        "hands_detected": hands_detected,
        "zero_frame": hands_detected == 0,
        "left": left,
        "right": right,
        "landmarks": {
            "left": arr[:63].reshape(21, 3).round(5).tolist() if left["present"] else None,
            "right": arr[63:].reshape(21, 3).round(5).tolist() if right["present"] else None,
        },
        "full_range": [_safe_float(arr.min()), _safe_float(arr.max())],
    }


def summarize_sequence(sequence, target_len):
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 126:
        return {"shape": list(arr.shape), "valid": False, "target_len": target_len}

    left = arr[:, :63]
    right = arr[:, 63:]
    left_present = np.max(np.abs(left), axis=1) > 1e-6
    right_present = np.max(np.abs(right), axis=1) > 1e-6
    any_present = left_present | right_present
    return {
        "shape": list(arr.shape),
        "valid": True,
        "target_len": target_len,
        "frames_ready": int(arr.shape[0]),
        "frames_with_any_hand": int(any_present.sum()),
        "zero_frames": int((~any_present).sum()),
        "left_present_frames": int(left_present.sum()),
        "right_present_frames": int(right_present.sum()),
        "left_missing_frames": int((~left_present).sum()),
        "right_missing_frames": int((~right_present).sum()),
    }


def summarize_preprocessed(input_tensor):
    arr = np.asarray(input_tensor, dtype=np.float32)
    finite = np.isfinite(arr)
    if arr.size == 0:
        return {"shape": list(arr.shape), "finite": True}

    return {
        "shape": list(arr.shape),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "min": _safe_float(np.nanmin(arr)),
        "max": _safe_float(np.nanmax(arr)),
        "mean": _safe_float(np.nanmean(arr)),
        "std": _safe_float(np.nanstd(arr)),
    }


# ═══════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════
def _signaturk_hand_debug_stats(frame, nodes):
    hand = np.asarray(frame, dtype=np.float32)[nodes]
    conf = hand[:, 3]
    xy_present = np.max(np.abs(hand[:, :2]), axis=1) > 1e-6
    visible = conf > 0
    present = bool(np.any(xy_present | visible))
    stats = {
        "present": present,
        "nonzero_ratio": _safe_float(np.count_nonzero(xy_present | visible) / len(nodes)),
        "confidence_mean": _safe_float(conf.mean()),
    }
    if not present:
        stats.update({"bbox": None, "range": None})
        return stats
    stats["bbox"] = {
        "x": [_safe_float(hand[:, 0].min()), _safe_float(hand[:, 0].max())],
        "y": [_safe_float(hand[:, 1].min()), _safe_float(hand[:, 1].max())],
        "z": [_safe_float(hand[:, 2].min()), _safe_float(hand[:, 2].max())],
    }
    stats["range"] = [_safe_float(hand[:, :3].min()), _safe_float(hand[:, :3].max())]
    return stats


def _display_filter_hand(frame, nodes, kpt_thr, min_points, min_mean_conf, max_span, max_area):
    hand = frame[nodes]
    present = hand[:, 3] >= kpt_thr
    if int(present.sum()) < min_points:
        frame[nodes] = 0.0
        return
    if float(hand[present, 3].mean()) < min_mean_conf:
        frame[nodes] = 0.0
        return
    xy = hand[present, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    if float(span[0]) > max_span or float(span[1]) > max_span:
        frame[nodes] = 0.0
        return
    if float(span[0] * span[1]) > max_area:
        frame[nodes] = 0.0


def filter_landmarks_for_display(landmarks):
    frame = np.asarray(landmarks, dtype=np.float32).reshape(ST_TOTAL_LANDMARKS, 4).copy()
    kpt_thr = float(os.getenv("SIGNATURK_OVERLAY_KPT_THR", "0.20"))
    min_points = int(os.getenv("SIGNATURK_OVERLAY_MIN_HAND_POINTS", "6"))
    min_mean = float(os.getenv("SIGNATURK_OVERLAY_MIN_HAND_MEAN_CONF", "0.20"))
    max_span = float(os.getenv("SIGNATURK_OVERLAY_MAX_HAND_BBOX_SPAN", "0.34"))
    max_area = float(os.getenv("SIGNATURK_OVERLAY_MAX_HAND_BBOX_AREA", "0.06"))
    _display_filter_hand(frame, ST_LEFT_HAND_NODES, kpt_thr, min_points, min_mean, max_span, max_area)
    _display_filter_hand(frame, ST_RIGHT_HAND_NODES, kpt_thr, min_points, min_mean, max_span, max_area)
    return frame


def filter_landmarks_for_model(landmarks):
    if os.getenv("SIGNATURK_FILTER_MODEL_HANDS", "true").lower() not in {"1", "true", "yes", "on"}:
        return np.asarray(landmarks, dtype=np.float32).reshape(ST_TOTAL_LANDMARKS, 4)

    frame = np.asarray(landmarks, dtype=np.float32).reshape(ST_TOTAL_LANDMARKS, 4).copy()
    kpt_thr = float(os.getenv("SIGNATURK_MODEL_KPT_THR", os.getenv("SIGNATURK_KPT_THR", "0.05")))
    min_points = int(os.getenv("SIGNATURK_MODEL_MIN_HAND_POINTS", os.getenv("SIGNATURK_MIN_HAND_POINTS", "6")))
    min_mean = float(os.getenv("SIGNATURK_MODEL_MIN_HAND_MEAN_CONF", os.getenv("SIGNATURK_MIN_HAND_MEAN_CONF", "0.20")))
    max_span = float(os.getenv("SIGNATURK_MODEL_MAX_HAND_BBOX_SPAN", os.getenv("SIGNATURK_MAX_HAND_BBOX_SPAN", "0.34")))
    max_area = float(os.getenv("SIGNATURK_MODEL_MAX_HAND_BBOX_AREA", os.getenv("SIGNATURK_MAX_HAND_BBOX_AREA", "0.06")))
    _display_filter_hand(frame, ST_LEFT_HAND_NODES, kpt_thr, min_points, min_mean, max_span, max_area)
    _display_filter_hand(frame, ST_RIGHT_HAND_NODES, kpt_thr, min_points, min_mean, max_span, max_area)
    return frame


def filter_sequence_for_display(sequence):
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, ST_TOTAL_LANDMARKS, 4)
    frames = arr.reshape(-1, ST_TOTAL_LANDMARKS, 4)
    return np.stack([filter_landmarks_for_display(frame) for frame in frames], axis=0)


def summarize_landmarks(landmarks):
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == ST_TOTAL_LANDMARKS * 4:
        frame = arr.reshape(ST_TOTAL_LANDMARKS, 4)
        left = _signaturk_hand_debug_stats(frame, ST_LEFT_HAND_NODES)
        right = _signaturk_hand_debug_stats(frame, ST_RIGHT_HAND_NODES)
        hands_detected = int(left["present"]) + int(right["present"])
        return {
            "shape": [ST_TOTAL_LANDMARKS, 4],
            "valid": True,
            "layout": "pose33+left_hand21+right_hand21 x xyzc",
            "hands_detected": hands_detected,
            "zero_frame": hands_detected == 0,
            "left": left,
            "right": right,
            "landmarks": {
                "left": frame[ST_LEFT_HAND_NODES, :3].round(5).tolist() if left["present"] else None,
                "right": frame[ST_RIGHT_HAND_NODES, :3].round(5).tolist() if right["present"] else None,
            },
            "full_range": [_safe_float(frame[:, :3].min()), _safe_float(frame[:, :3].max())],
        }
    return {"shape": list(arr.shape), "valid": False}


def summarize_sequence(sequence, target_len):
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.size and arr.shape[-2:] == (ST_TOTAL_LANDMARKS, 4):
        frames = arr.reshape(-1, ST_TOTAL_LANDMARKS, 4)
    elif arr.ndim == 2 and arr.shape[-1] == ST_TOTAL_LANDMARKS * 4:
        frames = arr.reshape(arr.shape[0], ST_TOTAL_LANDMARKS, 4)
    else:
        return {"shape": list(arr.shape), "valid": False, "target_len": target_len}

    left = frames[:, ST_LEFT_HAND_NODES, :]
    right = frames[:, ST_RIGHT_HAND_NODES, :]
    left_present = (left[:, :, 3].mean(axis=1) > 0) | (np.max(np.abs(left[:, :, :2]), axis=(1, 2)) > 1e-6)
    right_present = (right[:, :, 3].mean(axis=1) > 0) | (np.max(np.abs(right[:, :, :2]), axis=(1, 2)) > 1e-6)
    any_present = left_present | right_present
    return {
        "shape": list(frames.shape),
        "valid": True,
        "target_len": target_len,
        "frames_ready": int(frames.shape[0]),
        "frames_with_any_hand": int(any_present.sum()),
        "zero_frames": int((~any_present).sum()),
        "left_present_frames": int(left_present.sum()),
        "right_present_frames": int(right_present.sum()),
        "left_missing_frames": int((~left_present).sum()),
        "right_missing_frames": int((~right_present).sum()),
    }


def summarize_feature_bundle(features):
    arrays = list(features.skeleton_inputs) + list(features.hand_inputs)
    flat = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in arrays])
    finite = np.isfinite(flat)
    return {
        "shape": {
            "skeleton": [list(item.shape) for item in features.skeleton_inputs],
            "hand": [list(item.shape) for item in features.hand_inputs],
        },
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "min": _safe_float(np.nanmin(flat)),
        "max": _safe_float(np.nanmax(flat)),
        "mean": _safe_float(np.nanmean(flat)),
        "std": _safe_float(np.nanstd(flat)),
    }


def get_label(model_index: int) -> Dict[str, Any]:
    key = str(model_index)
    if key in LABEL_MAP:
        e = LABEL_MAP[key]
        return {"class_id": model_index,
                "label_tr": e.get("TR", f"class_{model_index}"),
                "label_en": e.get("EN", f"class_{model_index}")}
    return {"class_id": model_index,
            "label_tr": f"class_{model_index}",
            "label_en": f"class_{model_index}"}


def prediction_from_proba(proba_row):
    top_indices = np.argsort(proba_row)[::-1][:5]
    best_idx = int(top_indices[0])
    best_conf = float(proba_row[best_idx])
    best_lbl = get_label(best_idx)

    return {
        "class_id": best_lbl["class_id"],
        "label_tr": best_lbl["label_tr"],
        "label_en": best_lbl["label_en"],
        "confidence": best_conf,
        "animation_key": f"{best_lbl['label_en']}_sign",
        "category": "Sign",
        "above_threshold": best_conf >= CONFIDENCE_THRESHOLD,
        "top_predictions": [
            {"class_id": int(i), **get_label(int(i)),
             "confidence": round(float(proba_row[i]), 4)}
            for i in top_indices[:3]
        ],
    }


def run_inference(landmark_sequence=None, streams=None, features=None):
    global AVG_INFERENCE_MS, INFERENCE_COUNT
    if SIGNATURK_ENSEMBLE is None:
        return None
    if features is None:
        features = signaturk_build_feature_bundle(np.asarray(landmark_sequence, dtype=np.float32))
    prediction = SIGNATURK_ENSEMBLE.predict(features, streams=streams)
    elapsed_ms = prediction.latency_ms
    INFERENCE_COUNT += 1
    AVG_INFERENCE_MS = (AVG_INFERENCE_MS * (INFERENCE_COUNT-1) + elapsed_ms) / INFERENCE_COUNT
    return prediction_from_proba(prediction.probabilities)


def build_live_prediction_result(sequence, frame_debug, buffer_debug, base_timing, streams=None):
    t_job_start = time.perf_counter()
    orientation_mode = live_orientation_mode()
    variants = {}
    auto_selected = None
    if orientation_mode == "auto" or os.getenv("SIGNATURK_DEBUG_ORIENTATION_VARIANTS", "false").lower() in {"1", "true", "yes", "on"}:
        t_variants = time.perf_counter()
        variants, auto_selected = run_orientation_variants(sequence, streams=streams)
        base_timing = {
            **base_timing,
            "orientation_variants_ms": round((time.perf_counter() - t_variants) * 1000, 1),
        }

    applied_orientation = auto_selected if orientation_mode == "auto" and auto_selected else orientation_mode
    oriented_sequence, applied_orientation = transform_landmarks_by_mode(sequence, applied_orientation)

    t_features = time.perf_counter()
    features = signaturk_build_feature_bundle(np.asarray(oriented_sequence, dtype=np.float32))
    feature_ms = (time.perf_counter() - t_features) * 1000
    preprocess_debug = summarize_feature_bundle(features)

    t_model_start = time.perf_counter()
    prediction = run_inference(features=features, streams=streams)
    model_ms = (time.perf_counter() - t_model_start) * 1000
    if prediction is None:
        return None

    timing = {
        **base_timing,
        "feature_ms": round(feature_ms, 1),
            "model_ms": round(model_ms, 1),
            "prediction_job_ms": round((time.perf_counter() - t_job_start) * 1000, 1),
            "server_total_ms": round(
                float(base_timing.get("decode_ms", 0.0))
                + float(base_timing.get("mediapipe_ms", 0.0))
            + (time.perf_counter() - t_job_start) * 1000,
            1,
            ),
            "async_prediction": True,
            "streams": streams or SIGNATURK_REALTIME_STREAMS,
            "orientation_mode": orientation_mode,
            "orientation_applied": applied_orientation,
        }

    return {
        "class_id": prediction["class_id"],
        "label_tr": prediction["label_tr"] if prediction["above_threshold"]
                    else f"({prediction['label_tr']}?)",
        "label_en": prediction["label_en"],
        "confidence": prediction["confidence"],
        "animation_key": prediction["animation_key"],
        "category": prediction["category"],
        "top_predictions": prediction.get("top_predictions", []),
        "debug": {
            "frame": frame_debug,
            "buffer": buffer_debug,
            "preprocess": preprocess_debug,
            "timing": timing,
            "variants": variants,
        },
    }


POSE_LEFT_RIGHT_PAIRS = [
    (2, 5),    # eyes
    (7, 8),    # ears
    (11, 12),  # shoulders
    (13, 14),  # elbows
    (15, 16),  # wrists
    (23, 24),  # hips
    (25, 26),  # knees
    (27, 28),  # ankles
]
LIVE_ORIENTATION_MODES = {"normal", "mirror_x", "swap_hands", "mirror_x_swap", "auto"}


def live_orientation_mode():
    mode = os.getenv("SIGNATURK_LIVE_ORIENTATION", "normal").strip().lower()
    return mode if mode in LIVE_ORIENTATION_MODES else "normal"


def transform_landmark_sequence(sequence, swap_hands=False, mirror_x=False):
    original = np.asarray(sequence, dtype=np.float32)
    arr = original.copy().reshape(-1, ST_TOTAL_LANDMARKS, 4)
    if mirror_x:
        present = arr[:, :, 3] > 0
        arr[:, :, 0] = np.where(present, 1.0 - arr[:, :, 0], arr[:, :, 0])
    if swap_hands:
        for left_idx, right_idx in POSE_LEFT_RIGHT_PAIRS:
            arr[:, [left_idx, right_idx], :] = arr[:, [right_idx, left_idx], :]
        left_start = ST_POSE_COUNT
        right_start = ST_POSE_COUNT + ST_HAND_COUNT
        left = arr[:, left_start:left_start + ST_HAND_COUNT, :].copy()
        arr[:, left_start:left_start + ST_HAND_COUNT, :] = arr[:, right_start:right_start + ST_HAND_COUNT, :]
        arr[:, right_start:right_start + ST_HAND_COUNT, :] = left
    return arr.reshape(original.shape)


def transform_landmarks_by_mode(sequence, mode):
    if mode == "mirror_x":
        return transform_landmark_sequence(sequence, mirror_x=True), "mirror_x"
    if mode == "swap_hands":
        return transform_landmark_sequence(sequence, swap_hands=True), "swap_hands"
    if mode == "mirror_x_swap":
        return transform_landmark_sequence(sequence, mirror_x=True, swap_hands=True), "mirror_x_swap"
    return np.asarray(sequence, dtype=np.float32), "normal"


def run_orientation_variants(sequence, streams=None):
    variants = {}
    best_name = None
    best_confidence = -1.0
    for name in ["normal", "mirror_x", "swap_hands", "mirror_x_swap"]:
        variant_sequence, _ = transform_landmarks_by_mode(sequence, name)
        prediction = run_inference(variant_sequence, streams=streams)
        if prediction is None:
            continue
        variants[name] = {
            "class_id": prediction["class_id"],
            "label_tr": prediction["label_tr"],
            "label_en": prediction["label_en"],
            "confidence": prediction["confidence"],
            "top_predictions": prediction.get("top_predictions", []),
        }
        if prediction["confidence"] > best_confidence:
            best_confidence = prediction["confidence"]
            best_name = name
    return variants, best_name


# ═══════════════════════════════════════════════════════════
#  FRAME BUFFER
# ═══════════════════════════════════════════════════════════
class FrameBuffer:
    def __init__(self, seq_len=16):
        self.seq_len = seq_len
        self.min_frames = max(1, min(seq_len, int(os.getenv("SIGNATURK_LIVE_MIN_FRAMES", "6"))))
        self.window_seconds = max(0.25, float(os.getenv("SIGNATURK_LIVE_WINDOW_S", "2.2")))
        self.buffer: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.last_actual_frames = 0
        self.last_window_seconds = 0.0
        self.last_source = np.zeros((0, ST_TOTAL_LANDMARKS, 4), dtype=np.float32)

    def add(self, landmarks):
        now = time.perf_counter()
        self.buffer.append(landmarks)
        self.timestamps.append(now)
        elapsed = (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        has_enough_frames = len(self.buffer) >= self.min_frames
        reached_time_window = elapsed >= self.window_seconds
        reached_full_window = len(self.buffer) >= self.seq_len
        if not reached_full_window and (not has_enough_frames or not reached_time_window):
            return None
        actual = min(len(self.buffer), self.seq_len)
        source = self.buffer[:actual]
        source_array = np.stack(source, axis=0)
        if actual < self.seq_len:
            indices = signaturk_sample_indices(actual, self.seq_len)
            sequence = np.stack([source[int(idx)] for idx in indices], axis=0)
        else:
            sequence = source_array
        self.last_actual_frames = actual
        self.last_window_seconds = elapsed
        self.last_source = source_array
        self.buffer = self.buffer[actual:]
        self.timestamps = self.timestamps[actual:]
        return sequence

    def view(self):
        if not self.buffer:
            return np.zeros((0, ST_TOTAL_LANDMARKS, 4), dtype=np.float32)
        return np.stack(self.buffer, axis=0)

    def elapsed_seconds(self):
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
#  DB HELPER
# ═══════════════════════════════════════════════════════════
def add_log_db(db: Session, level: str, message: str, user: str = "system"):
    log = models.Log(level=level, message=message, user=user)
    db.add(log)
    db.commit()


def decode_base64_image(data_url):
    try:
        encoded = data_url.split(",", 1)[-1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"[DECODE] {e}")
        return None


# ── Schemas ───────────────────────────────────────────────
class SettingsData(BaseModel):
    camera: str
    voice: str
    speech_rate: float
    avatar_speed: float
    tts_enabled: bool = True
    notifications_enabled: bool = True
    avatar_enabled: bool = True
    websocket_enabled: bool = True

class RegisterData(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class LoginData(BaseModel):
    email: EmailStr
    password: str


# ═══════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, ACTIVE_DB_LABEL
    db_label = "Supabase PostgreSQL"
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        if os.environ.get("TSL_SQLITE_FALLBACK", "1").lower() not in {"1", "true", "yes", "on"}:
            raise
        sqlite_path = BASE_DIR / "local_dev.db"
        logger.warning(f"[DB] Supabase unavailable, using SQLite fallback: {exc}")
        engine = db_module.configure_database(f"sqlite:///{sqlite_path.as_posix()}")
        Base.metadata.create_all(bind=engine)
        db_label = f"SQLite fallback ({sqlite_path.name})"
    ACTIVE_DB_LABEL = db_label
    logger.info("[DB] Tablolar Supabase'de doğrulandı")

    logger.info(f"[DB] Active database: {db_label}")

    db = next(get_db())
    try:
        if db.query(models.User).count() == 0:
            db.add_all([
                models.User(full_name="Ayşe Kaya", email="ayse@tsl.ai",
                            password_hash=hash_pw("123456"), role="Admin",
                            status="Active", sessions=0),
                models.User(full_name="Mehmet Demir", email="mehmet@tsl.ai",
                            password_hash=hash_pw("123456"), role="User",
                            status="Active", sessions=0),
            ])
            db.commit()
            logger.info("[DB] Varsayılan kullanıcılar oluşturuldu")
        if db.query(models.Setting).count() == 0:
            db.add(models.Setting(id=1))
            db.commit()
        add_log_db(db, "Info", "TSL Nexus + Animasyon backend başlatıldı", "system")
    finally:
        db.close()

    load_model_assets()
    load_landmark_index()

    logger.info("=" * 60)
    logger.info("  TSL Nexus Backend v3.0 Hazır")
    logger.info(f"  Model:        {'YÜKLÜ' if MODEL else 'YOK'}")
    logger.info(f"  Landmark:     {len(LANDMARK_INDEX)} kelime")
    logger.info(f"  MediaPipe:    {'HAZIR' if MP_HANDS else 'EKSİK'}")
    logger.info(f"  Database:     {db_label}")
    logger.info("=" * 60)

    yield


app = FastAPI(title="TSL Nexus", version="3.0.0", lifespan=lifespan)

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="frontend")


# ═══════════════════════════════════════════════════════════
#  ROUTES — temel
# ═══════════════════════════════════════════════════════════
@app.get("/")
async def root():
    f = FRONTEND_DIR / "index.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error": "frontend/index.html bulunamadi."})


@app.get("/avatar3d")
async def avatar3d():
    f = FRONTEND_DIR / "avatar3d.html"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error": "frontend/avatar3d.html bulunamadi."})

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "mediapipe_ready": MP_HANDS is not None,
        "extractor": SIGNATURK_EXTRACTOR.status.__dict__ if SIGNATURK_EXTRACTOR is not None else None,
        "streams": SIGNATURK_REALTIME_STREAMS,
        "landmark_count": len(LANDMARK_INDEX),
        "database": ACTIVE_DB_LABEL,
    }


# ═══════════════════════════════════════════════════════════
#  ROUTES — 3D Animasyon (senin endpoint'lerin)
# ═══════════════════════════════════════════════════════════
@app.get("/signs")
async def list_signs():
    """Mevcut tüm işaret kelimelerini listele."""
    return {"words": sorted(LANDMARK_INDEX.keys()), "count": len(LANDMARK_INDEX)}

@app.get("/landmark/{word}")
async def get_landmark(word: str):
    """Kelimeye ait landmark verisini döndür (smooth edilmiş)."""
    key = word.lower().strip()
    data = LANDMARK_INDEX.get(key)
    if data is None:
        return Response(status_code=404, content=f"'{word}' bulunamadı")
    if SMOOTHER_AVAILABLE and not data.get("smoothed"):
        data = smooth_landmark_data(data)
    return Response(
        content=json.dumps(data, ensure_ascii=False),
        media_type="application/json"
    )

# ═══════════════════════════════════════════════════════════
#  ROUTES — Auth
# ═══════════════════════════════════════════════════════════
@app.post("/api/auth/register")
def register(data: RegisterData, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == data.email.lower()).first():
        return JSONResponse(status_code=400, content={"error": "Email zaten kayıtlı"})
    user = models.User(
        full_name=data.full_name, email=data.email.lower(),
        password_hash=hash_pw(data.password),
        role="User", status="Active", sessions=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    add_log_db(db, "Info", f"Yeni kullanıcı: {data.email}")
    return {"message": "Kayıt başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}

@app.post("/api/auth/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email.lower()).first()
    if not user or not verify_pw(data.password, user.password_hash):
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}")
    return {"message": "Giriş başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}


# ═══════════════════════════════════════════════════════════
#  ROUTES — Canlı tahmin WebSocket (tsl-nexus)
# ═══════════════════════════════════════════════════════════
@app.websocket("/api/predict/live-legacy")
async def live_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS-LIVE] Bağlandı")
    db = next(get_db())
    add_log_db(db, "Info", "Live WebSocket bağlandı")
    buf = FrameBuffer(seq_len=SEQ_LEN)
    capture_interval_ms = int(os.getenv("SIGNATURK_CAPTURE_INTERVAL_MS", "62"))
    stale_drain_limit = int(os.getenv("SIGNATURK_WS_DRAIN_LIMIT", "128"))
    prediction_cooldown_s = float(os.getenv("SIGNATURK_PREDICTION_COOLDOWN_S", "1.5"))
    prediction_cooldown_until = 0.0
    try:
        while True:
            t_receive = time.perf_counter()
            client_sent_at = None
            frame_id = None
            image_payload = None
            incoming = await websocket.receive_text()
            dropped_incoming = 0
            while dropped_incoming < stale_drain_limit:
                try:
                    incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    dropped_incoming += 1
                except asyncio.TimeoutError:
                    break
            t_received = time.perf_counter()
            landmarks = None
            try:
                payload = json.loads(incoming)
                client_sent_at = payload.get("sent_at")
                frame_id = payload.get("frame_id")
                if "landmarks" in payload:
                    arr = np.array(payload["landmarks"], dtype=np.float32)
                    if arr.size == ST_TOTAL_LANDMARKS * 4:
                        landmarks = arr.reshape(ST_TOTAL_LANDMARKS, 4)
                elif "image" in payload:
                    image_payload = payload["image"]
            except (json.JSONDecodeError, ValueError):
                pass

            if landmarks is None:
                t_decode_start = time.perf_counter()
                frame = decode_base64_image(image_payload or incoming)
                decode_ms = (time.perf_counter() - t_decode_start) * 1000
                if frame is None:
                    await websocket.send_json({"class_id": -1, "label_tr": "-",
                        "label_en": "-", "confidence": 0, "animation_key": "none"})
                    continue
                t_mp_start = time.perf_counter()
                if SIGNATURK_EXTRACTOR is None or not SIGNATURK_EXTRACTOR.status.available:
                    landmarks = None
                else:
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    landmarks = await asyncio.to_thread(SIGNATURK_EXTRACTOR.infer_frame, rgb)
                mediapipe_ms = (time.perf_counter() - t_mp_start) * 1000
                if landmarks is None:
                    await websocket.send_json({"class_id": -1,
                        "label_tr": "RTMPose kullanılamıyor",
                        "label_en": "Install rtmlib / onnxruntime",
                        "confidence": 0, "animation_key": "none"})
                    continue
            else:
                decode_ms = 0.0
                mediapipe_ms = 0.0

            landmarks = filter_landmarks_for_model(landmarks)
            frame_debug = summarize_landmarks(landmarks)
            cooldown_remaining_s = max(0.0, prediction_cooldown_until - time.perf_counter())
            if cooldown_remaining_s > 0:
                buffer_view = np.stack(buf.buffer, axis=0) if buf.buffer else np.zeros((0, ST_TOTAL_LANDMARKS, 4), dtype=np.float32)
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "Tahmin arası",
                    "label_en": "Cooldown",
                    "confidence": 0,
                    "animation_key": "none",
                    "debug": {
                        "frame": frame_debug,
                        "buffer": summarize_sequence(buffer_view, SEQ_LEN),
                        "timing": {
                            "frame_id": frame_id,
                            "capture_interval_ms": capture_interval_ms,
                            "dropped_incoming_frames": dropped_incoming,
                            "cooldown_remaining_s": round(cooldown_remaining_s, 2),
                            "receive_wait_ms": round((t_received - t_receive) * 1000, 1),
                            "decode_ms": round(decode_ms, 1),
                            "mediapipe_ms": round(mediapipe_ms, 1),
                            "model_ms": 0.0,
                            "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
                        },
                    },
                })
                continue
            sequence = buf.add(landmarks)
            buffer_view = sequence if sequence is not None else np.stack(buf.buffer, axis=0)
            buffer_debug = summarize_sequence(buffer_view, SEQ_LEN)
            base_timing = {
                "frame_id": frame_id,
                "capture_interval_ms": capture_interval_ms,
                "window_seconds": round(SEQ_LEN * capture_interval_ms / 1000.0, 2),
                "actual_frames_for_prediction": buf.last_actual_frames if sequence is not None else len(buf),
                "min_live_frames": buf.min_frames,
                "dropped_incoming_frames": dropped_incoming,
                "receive_wait_ms": round((t_received - t_receive) * 1000, 1),
                "decode_ms": round(decode_ms, 1),
                "mediapipe_ms": round(mediapipe_ms, 1),
            }
            if isinstance(client_sent_at, (int, float)):
                base_timing["client_to_server_ms"] = round(time.time() * 1000 - float(client_sent_at), 1)
            if sequence is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": f"Tampon ({len(buf)}/{SEQ_LEN})",
                    "label_en": "Collecting...",
                    "confidence": len(buf)/SEQ_LEN, "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": base_timing}})
                continue

            if MODEL is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": "Model yüklenmedi", "label_en": "Model not loaded",
                    "confidence": 0, "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": base_timing}})
                continue

            min_hand_frames = max(4, SEQ_LEN // 2)
            if buffer_debug.get("frames_with_any_hand", 0) < min_hand_frames:
                timing = {
                    **base_timing,
                    "model_ms": 0.0,
                    "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
                    "skipped_model": True,
                    "skip_reason": "not_enough_hand_frames",
                    "min_hand_frames": min_hand_frames,
                }
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "El bekleniyor",
                    "label_en": "Waiting for hand",
                    "confidence": 0,
                    "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": timing},
                })
                continue

            features = signaturk_build_feature_bundle(sequence)
            t_model_start = time.perf_counter()
            prediction = run_inference(features=features, streams=SIGNATURK_REALTIME_STREAMS or None)
            variants = {}
            if LIVE_VARIANTS_ENABLED:
                variants = {}
            model_ms = (time.perf_counter() - t_model_start) * 1000
            if prediction is None:
                continue
            preprocess_debug = summarize_feature_bundle(features)
            timing = {
                **base_timing,
                "model_ms": round(model_ms, 1),
                "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
            }

            result = {
                "class_id": prediction["class_id"],
                "label_tr": prediction["label_tr"] if prediction["above_threshold"]
                            else f"({prediction['label_tr']}?)",
                "label_en": prediction["label_en"],
                "confidence": prediction["confidence"],
                "animation_key": prediction["animation_key"],
                "category": prediction["category"],
                "top_predictions": prediction.get("top_predictions", []),
                "debug": {
                    "frame": frame_debug,
                    "buffer": buffer_debug,
                    "preprocess": preprocess_debug,
                    "timing": timing,
                    "variants": variants,
                },
            }

            if prediction["above_threshold"]:
                db.add(models.History(
                    session_id=uuid.uuid4().hex[:8], mode="Live",
                    result=prediction["label_tr"],
                    confidence=f"{prediction['confidence']*100:.1f}%",
                ))
                db.commit()

            await websocket.send_json(result)
            prediction_cooldown_until = time.perf_counter() + prediction_cooldown_s

    except WebSocketDisconnect:
        logger.info("[WS-LIVE] Bağlantı kesildi")
        add_log_db(db, "Info", "Live WebSocket kesildi")
    except Exception as e:
        logger.error(f"[WS-LIVE] Hata: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"WebSocket hatası: {e}")
    finally:
        db.close()


@app.websocket("/api/predict/live")
async def live_predict_async(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS-LIVE-ASYNC] Baglandi")
    db = next(get_db())
    add_log_db(db, "Info", "Live WebSocket baglandi")

    buf = FrameBuffer(seq_len=SEQ_LEN)
    capture_interval_ms = int(os.getenv("SIGNATURK_CAPTURE_INTERVAL_MS", "62"))
    stale_drain_limit = int(os.getenv("SIGNATURK_WS_DRAIN_LIMIT", "128"))
    prediction_cooldown_s = float(os.getenv("SIGNATURK_PREDICTION_COOLDOWN_S", "0.8"))
    prediction_cooldown_until = 0.0
    prediction_task = None
    pending_prediction = None

    def start_prediction(job):
        nonlocal prediction_task, pending_prediction
        sequence, frame_debug, buffer_debug, base_timing = job
        prediction_task = asyncio.create_task(asyncio.to_thread(
            build_live_prediction_result,
            sequence,
            frame_debug,
            buffer_debug,
            base_timing,
            SIGNATURK_REALTIME_STREAMS or None,
        ))
        pending_prediction = None

    async def flush_prediction_if_ready():
        nonlocal prediction_task, prediction_cooldown_until
        if prediction_task is None or not prediction_task.done():
            return False

        try:
            result = prediction_task.result()
        except Exception as exc:
            logger.error(f"[WS-LIVE-ASYNC] Prediction failed: {exc}\n{traceback.format_exc()}")
            result = {
                "class_id": -1,
                "label_tr": "Tahmin hatasi",
                "label_en": "Prediction error",
                "confidence": 0,
                "animation_key": "none",
                "debug": {"timing": {"async_prediction": True, "error": str(exc)}},
            }

        prediction_task = None
        prediction_cooldown_until = time.perf_counter() + prediction_cooldown_s

        if result is not None and result.get("class_id", -1) != -1:
            confidence = float(result.get("confidence", 0.0))
            label_tr = str(result.get("label_tr", ""))
            if confidence >= CONFIDENCE_THRESHOLD and label_tr:
                db.add(models.History(
                    session_id=uuid.uuid4().hex[:8],
                    mode="Live",
                    result=label_tr.replace("(", "").replace("?)", ""),
                    confidence=f"{confidence*100:.1f}%",
                ))
                db.commit()

        if result is not None:
            await websocket.send_json(result)
        return True

    def maybe_start_pending():
        nonlocal pending_prediction
        if prediction_task is None and pending_prediction is not None:
            if time.perf_counter() >= prediction_cooldown_until:
                start_prediction(pending_prediction)
                return True
        return False

    try:
        while True:
            t_receive = time.perf_counter()
            client_sent_at = None
            frame_id = None
            image_payload = None
            incoming = await websocket.receive_text()

            dropped_incoming = 0
            while dropped_incoming < stale_drain_limit:
                try:
                    incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    dropped_incoming += 1
                except asyncio.TimeoutError:
                    break

            t_received = time.perf_counter()
            received_wall_ms = time.time() * 1000
            landmarks = None
            try:
                payload = json.loads(incoming)
                client_sent_at = payload.get("sent_at")
                frame_id = payload.get("frame_id")
                if "landmarks" in payload:
                    arr = np.array(payload["landmarks"], dtype=np.float32)
                    if arr.size == ST_TOTAL_LANDMARKS * 4:
                        landmarks = arr.reshape(ST_TOTAL_LANDMARKS, 4)
                elif "image" in payload:
                    image_payload = payload["image"]
            except (json.JSONDecodeError, ValueError):
                pass

            if landmarks is None:
                t_decode_start = time.perf_counter()
                frame = decode_base64_image(image_payload or incoming)
                decode_ms = (time.perf_counter() - t_decode_start) * 1000
                if frame is None:
                    await websocket.send_json({
                        "class_id": -1,
                        "label_tr": "-",
                        "label_en": "-",
                        "confidence": 0,
                        "animation_key": "none",
                    })
                    continue

                t_mp_start = time.perf_counter()
                if SIGNATURK_EXTRACTOR is None or not SIGNATURK_EXTRACTOR.status.available:
                    landmarks = None
                else:
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    landmarks = await asyncio.to_thread(SIGNATURK_EXTRACTOR.infer_frame, rgb)
                mediapipe_ms = (time.perf_counter() - t_mp_start) * 1000

                if landmarks is None:
                    await websocket.send_json({
                        "class_id": -1,
                        "label_tr": "RTMPose kullanilamiyor",
                        "label_en": "Extractor unavailable",
                        "confidence": 0,
                        "animation_key": "none",
                    })
                    continue
            else:
                decode_ms = 0.0
                mediapipe_ms = 0.0

            landmarks = filter_landmarks_for_model(landmarks)
            await flush_prediction_if_ready()
            maybe_start_pending()

            display_landmarks = filter_landmarks_for_display(landmarks)
            frame_debug = summarize_landmarks(display_landmarks)
            sequence = buf.add(landmarks)
            buffer_view = sequence if sequence is not None else buf.view()
            buffer_view_display = filter_sequence_for_display(buffer_view)
            buffer_debug = summarize_sequence(buffer_view_display, SEQ_LEN)
            base_timing = {
                "frame_id": frame_id,
                "capture_interval_ms": capture_interval_ms,
                "window_seconds": round(buf.last_window_seconds if sequence is not None else buf.elapsed_seconds(), 2),
                "target_window_seconds": round(buf.window_seconds, 2),
                "actual_frames_for_prediction": buf.last_actual_frames if sequence is not None else len(buf),
                "actual_hand_frames": int(summarize_sequence(filter_sequence_for_display(buf.last_source), SEQ_LEN).get("frames_with_any_hand", 0)) if sequence is not None else int(buffer_debug.get("frames_with_any_hand", 0)),
                "model_input_frames": SEQ_LEN,
                "min_live_frames": buf.min_frames,
                "dropped_incoming_frames": dropped_incoming,
                "prediction_running": prediction_task is not None,
                "pending_prediction": pending_prediction is not None,
                "cooldown_remaining_s": round(max(0.0, prediction_cooldown_until - time.perf_counter()), 2),
                "receive_wait_ms": round((t_received - t_receive) * 1000, 1),
                "decode_ms": round(decode_ms, 1),
                "mediapipe_ms": round(mediapipe_ms, 1),
                "model_ms": 0.0,
                "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
                "streams": SIGNATURK_REALTIME_STREAMS,
            }
            if isinstance(client_sent_at, (int, float)):
                base_timing["client_to_server_ms"] = round(received_wall_ms - float(client_sent_at), 1)
                base_timing["landmark_age_ms"] = round(time.time() * 1000 - float(client_sent_at), 1)

            if sequence is None:
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "Tahmin hazirlaniyor" if prediction_task is not None else f"Tampon ({len(buf)}/{buf.min_frames})",
                    "label_en": "Predicting..." if prediction_task is not None else "Collecting...",
                    "confidence": min(1.0, len(buf) / max(buf.min_frames, 1)),
                    "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": base_timing},
                })
                await flush_prediction_if_ready()
                maybe_start_pending()
                continue

            if MODEL is None:
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "Model yuklenmedi",
                    "label_en": "Model not loaded",
                    "confidence": 0,
                    "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": base_timing},
                })
                continue

            real_buffer_debug = summarize_sequence(filter_sequence_for_display(buf.last_source), SEQ_LEN)
            min_hand_frames = max(3, min(buf.min_frames, max(1, buf.min_frames // 2)))
            real_hand_frames = int(real_buffer_debug.get("frames_with_any_hand", 0))
            if real_hand_frames < min_hand_frames:
                timing = {
                    **base_timing,
                    "skipped_model": True,
                    "skip_reason": "not_enough_hand_frames",
                    "min_hand_frames": min_hand_frames,
                    "actual_hand_frames": real_hand_frames,
                    "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
                }
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "El bekleniyor",
                    "label_en": "Waiting for hand",
                    "confidence": 0,
                    "animation_key": "none",
                    "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": timing},
                })
                await flush_prediction_if_ready()
                maybe_start_pending()
                continue

            job = (np.asarray(sequence, dtype=np.float32).copy(), frame_debug, buffer_debug, dict(base_timing))
            if prediction_task is None and time.perf_counter() >= prediction_cooldown_until:
                start_prediction(job)
                label_tr = "Tahmin hazirlaniyor"
                label_en = "Predicting..."
            else:
                pending_prediction = job
                label_tr = "Siradaki hareket hazir"
                label_en = "Queued..."

            status_timing = {
                **base_timing,
                "prediction_running": prediction_task is not None,
                "pending_prediction": pending_prediction is not None,
                "server_total_ms": round((time.perf_counter() - t_received) * 1000, 1),
            }
            await websocket.send_json({
                "class_id": -1,
                "label_tr": label_tr,
                "label_en": label_en,
                "confidence": 0,
                "animation_key": "none",
                "debug": {"frame": frame_debug, "buffer": buffer_debug, "timing": status_timing},
            })
            await flush_prediction_if_ready()
            maybe_start_pending()

    except WebSocketDisconnect:
        logger.info("[WS-LIVE-ASYNC] Baglanti kesildi")
        add_log_db(db, "Info", "Live WebSocket kesildi")
    except Exception as e:
        logger.error(f"[WS-LIVE-ASYNC] Hata: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"WebSocket hatasi: {e}")
    finally:
        if prediction_task is not None and not prediction_task.done():
            prediction_task.cancel()
        db.close()


@app.post("/api/predict/sequence")
async def predict_sequence(payload: Dict[str, Any]):
    try:
        raw = np.array(payload["landmarks"], dtype=np.float32)
        if raw.shape == (SEQ_LEN, ST_TOTAL_LANDMARKS * 4):
            raw = raw.reshape(SEQ_LEN, ST_TOTAL_LANDMARKS, 4)
        if raw.shape != (SEQ_LEN, ST_TOTAL_LANDMARKS, 4):
            return JSONResponse(status_code=400,
                content={"error": f"Beklenen ({SEQ_LEN}, {ST_TOTAL_LANDMARKS}, 4), gelen {list(raw.shape)}"})
        result = run_inference(raw)
        if result is None:
            return JSONResponse(status_code=503, content={"error": "Model yüklenmedi"})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════
#  ROUTES — History / Settings / Dictionary / Admin / Avatar
# ═══════════════════════════════════════════════════════════
@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    return [{"id": r.session_id, "mode": r.mode, "result": r.result,
             "conf": r.confidence,
             "time": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else ""}
            for r in rows]

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db)):
    s = db.query(models.Setting).first()
    if not s:
        return {"camera": "Default Camera", "voice": "Female Voice",
                "speech_rate": 1.0, "avatar_speed": 1.0,
                "tts_enabled": True, "notifications_enabled": True,
                "avatar_enabled": True, "websocket_enabled": True}
    return {"camera": s.camera, "voice": s.voice,
            "speech_rate": s.speech_rate, "avatar_speed": s.avatar_speed,
            "tts_enabled": s.tts_enabled, "notifications_enabled": s.notifications_enabled,
            "avatar_enabled": s.avatar_enabled, "websocket_enabled": s.websocket_enabled}

@app.post("/api/settings")
def save_settings(data: SettingsData, db: Session = Depends(get_db)):
    s = db.query(models.Setting).first()
    if not s:
        s = models.Setting(id=1)
        db.add(s)
    for field, val in data.dict().items():
        setattr(s, field, val)
    db.commit()
    add_log_db(db, "Info", "Ayarlar güncellendi", "admin")
    return {"message": "Ayarlar kaydedildi", "data": data.dict()}

@app.get("/api/dictionary")
async def get_dictionary():
    return [{"classId": int(k), "tr": v.get("TR", "?"), "en": v.get("EN", "?"),
             "animation": f"{v.get('EN','unknown')}_sign", "category": "Sign"}
            for k, v in sorted(LABEL_MAP.items(), key=lambda x: int(x[0]))]

@app.get("/api/dictionary/{class_id}")
async def get_dictionary_by_id(class_id: int):
    info = get_label(class_id)
    return {"classId": info["class_id"], "tr": info["label_tr"],
            "en": info["label_en"], "animation": f"{info['label_en']}_sign"}

@app.get("/api/avatar/{class_id}")
async def get_avatar(class_id: int):
    info = get_label(class_id)
    return {"class_id": info["class_id"], "label_tr": info["label_tr"],
            "label_en": info["label_en"], "animation_key": f"{info['label_en']}_sign"}

@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    confs = []
    for r in rows:
        try:
            confs.append(float(r.confidence.replace("%", "")))
        except:
            pass
    return {
        "total_users": db.query(models.User).count(),
        "active_users": db.query(models.User).filter(models.User.status == "Active").count(),
        "translations": db.query(models.History).count(),
        "avg_confidence": round(sum(confs)/len(confs), 1) if confs else 0.0,
    }

@app.get("/api/admin/users")
def admin_users(db: Session = Depends(get_db)):
    return [{"name": u.full_name, "email": u.email,
             "role": u.role, "sessions": u.sessions, "status": u.status,
             "joined": u.created_at.strftime("%Y-%m-%d") if u.created_at else "—"}
            for u in db.query(models.User).order_by(models.User.created_at.desc()).all()]

@app.get("/api/admin/stats")
def admin_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func as sqlfunc
    top = (db.query(models.History.result, sqlfunc.count(models.History.id).label("cnt"))
           .group_by(models.History.result)
           .order_by(sqlfunc.count(models.History.id).desc())
           .limit(12).all())
    total = db.query(models.History).count()
    return {
        "top_words": [{"word": r.result, "count": r.cnt} for r in top],
        "total_predictions": total,
    }

@app.get("/api/admin/logs")
def admin_logs(db: Session = Depends(get_db)):
    return [{"level": l.level, "message": l.message, "user": l.user,
             "when": l.created_at.strftime("%Y-%m-%d %H:%M") if l.created_at else ""}
            for l in db.query(models.Log).order_by(
                models.Log.created_at.desc()).limit(100).all()]

def _ensemble_model_shapes():
    if SIGNATURK_ENSEMBLE is None:
        return {}
    shapes = {}
    for name, model in SIGNATURK_ENSEMBLE.models.items():
        shapes[name] = {
            "input": [list(item.shape) for item in model.inputs],
            "output": [list(item.shape) for item in model.outputs],
        }
    return shapes


@app.get("/api/admin/model")
async def admin_model():
    return {
        "model": "LOADED ✓" if MODEL else "NOT LOADED ✗",
        "num_classes": NUM_CLASSES,
        "landmark_words": len(LANDMARK_INDEX),
        "mediapipe_ready": MP_HANDS is not None,
        "extractor": SIGNATURK_EXTRACTOR.status.__dict__ if SIGNATURK_EXTRACTOR is not None else None,
        "streams": SIGNATURK_REALTIME_STREAMS,
        "weights": SIGNATURK_ENSEMBLE.weights if SIGNATURK_ENSEMBLE is not None else {},
        "model_shapes": _ensemble_model_shapes(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "avg_inference": f"{AVG_INFERENCE_MS:.0f} ms" if INFERENCE_COUNT else "Henüz yok",
        "database": ACTIVE_DB_LABEL,
    }

@app.get("/api/debug/model-check")
async def debug_model_check():
    return {
        "model_loaded": MODEL is not None,
        "model_shapes": _ensemble_model_shapes(),
        "mediapipe_ready": MP_HANDS is not None,
        "extractor": SIGNATURK_EXTRACTOR.status.__dict__ if SIGNATURK_EXTRACTOR is not None else None,
        "streams": SIGNATURK_REALTIME_STREAMS,
        "label_map_entries": len(LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "norm_stats_loaded": NORM_MEAN is not None,
        "norm_mean_shape": list(NORM_MEAN.shape) if NORM_MEAN is not None else None,
        "norm_std_shape": list(NORM_STD.shape) if NORM_STD is not None else None,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "num_classes": NUM_CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "smoother_available": SMOOTHER_AVAILABLE,
        "config": DEMO_CONFIG,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
