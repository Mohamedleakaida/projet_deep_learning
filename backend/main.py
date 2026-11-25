"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PROTOTYPE D'AIDE À LA VISUALISATION – ONCOLOGIE CÉRÉBRALE          ║
║          FastAPI Backend  ·  Critères Bastien & Scapin intégrés             ║
╚══════════════════════════════════════════════════════════════════════════════╝

LANCEMENT :
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000
    # → Ouvrez http://localhost:8000

AJUSTEMENTS MODÈLES :
    • CLASSIFIER_INPUT_SIZE  : (H, W) attendus par EfficientNetB0
    • SEGMENTATION_INPUT_SIZE: (H, W) attendus par TransUNet
    • CLASS_NAMES            : ordre exact des neurones de sortie
    • NORMAL_LABELS          : sous-ensemble de CLASS_NAMES = "pas de cancer"
    • NORMAL_THRESHOLD       : seuil de confiance minimum pour déclarer "normal"
"""

from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  ← AJUSTEZ ICI SELON VOS MODÈLES
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFIER_INPUT_SIZE   = (224, 224)      # (H, W) pour EfficientNetB0
SEGMENTATION_INPUT_SIZE = (224, 224)      # (H, W) pour TransUNet

# Noms de classes dans l'ORDRE EXACT des sorties du classificateur
CLASS_NAMES = ["Glioma", "Meningioma", "Normal (No Tumor)", "Pituitary"]

# Libellés considérés comme "pas de cancer" → PAS de segmentation
NORMAL_LABELS: set[str] = {"Normal (No Tumor)"}

# Si la confiance de la classe prédite est < ce seuil, on affiche "Incertain"
CONFIDENCE_THRESHOLD = 0.50

# Chemins modèles (relatifs au répertoire de travail = racine du projet)
BASE_DIR        = Path(__file__).resolve().parent.parent
MODEL_DIR       = BASE_DIR                               # modèles à la racine
CLASSIFIER_PATH = MODEL_DIR / "EfficientNetB0.keras"
SEGMENTATION_PATH = MODEL_DIR / "TransUNet_finetuned.keras"

# ──────────────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT DES MODÈLES (avec mode MOCK si Keras absent ou fichiers manquants)
# ──────────────────────────────────────────────────────────────────────────────

keras_available = False
classifier_model = None
segmentation_model = None
MOCK_MODE = False

def _try_load_keras() -> None:
    """Tente d'importer Keras/TensorFlow et de charger les modèles."""
    global keras_available, classifier_model, segmentation_model, MOCK_MODE

    try:
        # Support Keras 2 (via TF) ou Keras 3 autonome
        try:
            import tensorflow as tf                        # noqa: F401
            from tensorflow import keras                   # noqa: F401
            load_model = tf.keras.models.load_model
        except ImportError:
            import keras                                   # noqa: F401
            from keras.models import load_model            # type: ignore

        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Classificateur introuvable : {CLASSIFIER_PATH}")
        if not SEGMENTATION_PATH.exists():
            raise FileNotFoundError(f"Segmentation introuvable : {SEGMENTATION_PATH}")

        print("[INFO] Chargement du classificateur…", flush=True)
        classifier_model = load_model(str(CLASSIFIER_PATH), compile=False)
        print(f"       ✓ input={classifier_model.input_shape}  "
              f"output={classifier_model.output_shape}", flush=True)

        print("[INFO] Chargement du modèle de segmentation…", flush=True)
        segmentation_model = load_model(str(SEGMENTATION_PATH), compile=False)
        print(f"       ✓ input={segmentation_model.input_shape}  "
              f"output={segmentation_model.output_shape}", flush=True)

        keras_available = True
        print("[INFO] ✅ Modèles chargés – mode PRODUCTION actif.", flush=True)

    except Exception as exc:
        MOCK_MODE = True
        print(f"[WARN] Modèles non chargés ({exc}). "
              "→ MODE MOCK activé (résultats simulés).", flush=True)


_try_load_keras()

# ──────────────────────────────────────────────────────────────────────────────
# 3. APPLICATION FASTAPI
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroScan Vision – Prototype d'aide à la visualisation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert le frontend statique
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/status")
async def status():
    """Critère Guidage (B&S 1) : état du serveur visible côté UI."""
    return {
        "mode": "mock" if MOCK_MODE else "production",
        "classifier": str(CLASSIFIER_PATH) if not MOCK_MODE else None,
        "segmentation": str(SEGMENTATION_PATH) if not MOCK_MODE else None,
        "classes": CLASS_NAMES,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. FONCTIONS DE PRÉTRAITEMENT
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_for_classifier(img: Image.Image) -> np.ndarray:
    """
    Redimensionne + normalise pour EfficientNetB0.
    EfficientNetB0 attend des pixels [0, 255] en float32
    (prétraitement interne via tf.keras.applications.efficientnet.preprocess_input).
    Ajustez la normalisation ici si votre pipeline est différent.
    """
    img_rgb = img.convert("RGB").resize(
        (CLASSIFIER_INPUT_SIZE[1], CLASSIFIER_INPUT_SIZE[0]),
        Image.LANCZOS,
    )
    arr = np.array(img_rgb, dtype=np.float32)        # [0, 255]
    # Si votre modèle attend [0, 1] décommentez :
    # arr = arr / 255.0
    # Si votre modèle utilise tf.keras.applications.efficientnet.preprocess_input :
    # from tensorflow.keras.applications.efficientnet import preprocess_input
    # arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)               # (1, H, W, 3)


def preprocess_for_segmentation(img: Image.Image) -> np.ndarray:
    """
    Redimensionne + normalise pour TransUNet.
    TransUNet attend généralement des pixels [0, 1].
    """
    img_rgb = img.convert("RGB").resize(
        (SEGMENTATION_INPUT_SIZE[1], SEGMENTATION_INPUT_SIZE[0]),
        Image.LANCZOS,
    )
    arr = np.array(img_rgb, dtype=np.float32) / 255.0   # [0, 1]
    return np.expand_dims(arr, axis=0)                   # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 5. FONCTIONS MOCK (simulation quand Keras absent)
# ──────────────────────────────────────────────────────────────────────────────

_rng = np.random.default_rng(seed=42)


def mock_classify(filename: str) -> dict[str, Any]:
    """
    Simule la classification à partir du nom de fichier.
    Préfixes reconnus : noTr→Normal, glTr→Glioma, meTr→Meningioma, piTr→Pituitary
    """
    fn = filename.lower()
    if "notr" in fn:
        probs = [0.04, 0.03, 0.91, 0.02]
    elif "gltr" in fn:
        probs = [0.85, 0.07, 0.03, 0.05]
    elif "metr" in fn:
        probs = [0.06, 0.88, 0.04, 0.02]
    elif "pitr" in fn:
        probs = [0.05, 0.04, 0.03, 0.88]
    else:
        probs = list(_rng.dirichlet(np.ones(len(CLASS_NAMES))))
    return {i: float(p) for i, p in enumerate(probs)}


def mock_segment(img: Image.Image) -> np.ndarray:
    """Génère un masque de segmentation simulé (ellipse centrée)."""
    h, w = SEGMENTATION_INPUT_SIZE
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    ry, rx = h // 5, w // 5
    Y, X = np.ogrid[:h, :w]
    ellipse = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
    mask[ellipse <= 1.0] = 0.90
    mask[np.logical_and(ellipse > 1.0, ellipse <= 1.3)] = 0.40
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# 6. FONCTIONS PRODUCTION (inférence réelle)
# ──────────────────────────────────────────────────────────────────────────────

def real_classify(img: Image.Image) -> dict[str, Any]:
    arr = preprocess_for_classifier(img)
    preds = classifier_model.predict(arr, verbose=0)[0]      # shape (n_classes,)
    return {i: float(p) for i, p in enumerate(preds)}


def real_segment(img: Image.Image) -> np.ndarray:
    arr = preprocess_for_segmentation(img)
    output = segmentation_model.predict(arr, verbose=0)[0]   # shape (H, W, C) ou (H, W)
    if output.ndim == 3:
        # Prend le canal de probabilité "tumeur" (souvent le dernier ou celui à l'index 1)
        # Ajustez l'index si votre modèle a un ordre différent
        mask = output[..., -1] if output.shape[-1] > 1 else output[..., 0]
    else:
        mask = output
    return mask.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 7. HELPER : MASQUE → IMAGE SUPERPOSÉE EN BASE64
# ──────────────────────────────────────────────────────────────────────────────

def build_overlay(original: Image.Image, mask_arr: np.ndarray,
                  alpha: float = 0.45) -> str:
    """
    Superpose le masque de segmentation (chaleur rouge) sur l'image originale.
    Retourne un PNG encodé en base64.
    """
    h, w = SEGMENTATION_INPUT_SIZE
    orig_resized = original.convert("RGBA").resize((w, h), Image.LANCZOS)

    # Masque → colormap thermique (rouge = forte probabilité tumorale)
    mask_norm = np.clip(mask_arr, 0, 1)
    r = (mask_norm * 255).astype(np.uint8)
    g = ((1 - mask_norm) * 80).astype(np.uint8)
    b = np.zeros_like(r)
    a = (mask_norm * 200 + 30).astype(np.uint8)   # transparence proportionnelle
    overlay_arr = np.stack([r, g, b, a], axis=-1)
    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")

    combined = Image.alpha_composite(orig_resized, overlay_img)
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# 8. ENDPOINT PRINCIPAL /analyze
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """
    Pipeline complet :
      1. Réception image
      2. Classification (mock ou production)
      3. Si cancer → Segmentation
      4. Construction réponse JSON

    Critères B&S appliqués :
      - Guidage (1) : chaque étape est loggée côté serveur
      - Charge de travail (2) : un seul endpoint, réponse JSON normalisée
      - Gestion des erreurs (5) : try/except avec messages explicites
      - Signifiance (7) : labels lisibles + probabilités
    """
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "image_unreadable",
                "message": "L'image n'a pas pu être lue. Vérifiez que le fichier n'est pas corrompu.",
            },
        )

    filename = file.filename or "image"

    # ── ÉTAPE 2 : CLASSIFICATION ──────────────────────────────────────────────
    try:
        if MOCK_MODE:
            class_probs = mock_classify(filename)
        else:
            class_probs = real_classify(img)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "classifier_failed",
                "message": f"Erreur lors de la classification : {exc}",
            },
        )

    idx_best   = max(class_probs, key=class_probs.__getitem__)
    label_best = CLASS_NAMES[idx_best]
    conf_best  = class_probs[idx_best]
    is_normal  = label_best in NORMAL_LABELS and conf_best >= CONFIDENCE_THRESHOLD
    uncertain  = conf_best < CONFIDENCE_THRESHOLD

    scores = [
        {"class": CLASS_NAMES[i], "probability": round(class_probs[i] * 100, 1)}
        for i in range(len(CLASS_NAMES))
    ]
    scores.sort(key=lambda x: x["probability"], reverse=True)

    # Image originale encodée (affichage côté UI)
    orig_b64 = image_to_b64(img)

    # ── ÉTAPE 3 : SEGMENTATION CONDITIONNELLE ────────────────────────────────
    overlay_b64 = None
    mask_confidence = None

    if not is_normal:
        try:
            if MOCK_MODE:
                mask = mock_segment(img)
            else:
                mask = real_segment(img)

            overlay_b64     = build_overlay(img, mask)
            mask_confidence = round(float(mask.mean()) * 100, 1)
        except Exception as exc:
            # Dégradation gracieuse : on garde la classification même si la
            # segmentation échoue (Critère B&S 5 – gestion des erreurs)
            overlay_b64     = None
            mask_confidence = None
            print(f"[WARN] Segmentation échouée : {exc}", flush=True)

    # ── RÉPONSE ──────────────────────────────────────────────────────────────
    return JSONResponse({
        "mock_mode":        MOCK_MODE,
        "filename":         filename,
        "predicted_class":  label_best,
        "confidence":       round(conf_best * 100, 1),
        "uncertain":        uncertain,
        "is_normal":        is_normal,
        "scores":           scores,
        "original_image":   orig_b64,
        "overlay_image":    overlay_b64,          # None si cas normal ou erreur
        "mask_confidence":  mask_confidence,
    })
