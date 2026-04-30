from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resp_ai.config import load_app_config, load_yaml
from resp_ai.inference.predictor import Predictor
from resp_ai.paths import resolve_project_path

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_strong_cnn_pneumonia_focus.yaml"
CONFIG_PATH = Path(os.environ.get("RESP_AI_CONFIG", DEFAULT_CONFIG_PATH)).expanduser().resolve()
RAW_CONFIG = load_yaml(CONFIG_PATH)
APP_CONFIG = load_app_config(RAW_CONFIG)

app = Flask(__name__)
CORS(app)
_predictor: Predictor | None = None
_predictor_load_error: str | None = None


@app.get("/")
def index() -> tuple[dict, int]:
    return {
        "service": "Respiratory AI Backend",
        "status": "online",
        "message": "Use /ready to verify the model, /health for service metadata, and /predict to analyze an audio file.",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
        },
    }, 200


def resolve_active_model_path() -> Path:
    model_path = os.environ.get("RESP_AI_MODEL")
    if model_path:
        return Path(model_path).expanduser().resolve()
    return (resolve_project_path(PROJECT_ROOT, RAW_CONFIG["paths"]["models_root"]) / "latest" / "best_model.keras").resolve()


def get_predictor() -> Predictor:
    global _predictor, _predictor_load_error
    if _predictor is None:
        try:
            model_path = resolve_active_model_path()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}. Copy the best Colab checkpoint into this path "
                    "or set RESP_AI_MODEL to the exported best_model.keras location."
                )
            _predictor = Predictor(config_path=CONFIG_PATH, model_path=model_path)
            _predictor_load_error = None
        except Exception as exc:
            _predictor_load_error = str(exc)
            raise
    return _predictor


def predictor_ready() -> tuple[bool, str | None]:
    if _predictor is not None:
        return True, None

    try:
        get_predictor()
        return True, None
    except Exception as exc:
        return False, str(exc)


@app.get("/health")
def health() -> tuple[dict, int]:
    active_model_path = resolve_active_model_path()
    inference_settings = RAW_CONFIG.get("inference", {})
    ready = _predictor is not None and _predictor_load_error is None
    return {
        "status": "ok",
        "ready": ready,
        "model_loaded": _predictor is not None,
        "model_error": _predictor_load_error,
        "model_path": str(active_model_path),
        "model_exists": active_model_path.exists(),
        "config_path": str(CONFIG_PATH),
        "aggregation": inference_settings.get("aggregation", "mean_probability"),
        "window_overlap": inference_settings.get("window_overlap", 0.5),
        "max_windows": inference_settings.get("max_windows", 5),
    }, 200


@app.get("/ready")
def ready() -> tuple[dict, int]:
    active_model_path = resolve_active_model_path()
    is_ready, error = predictor_ready()
    payload = {
        "status": "ok" if is_ready else "error",
        "ready": is_ready,
        "model_loaded": _predictor is not None,
        "model_path": str(active_model_path),
        "model_exists": active_model_path.exists(),
        "config_path": str(CONFIG_PATH),
        "error": error,
    }
    return payload, 200 if is_ready else 503


if os.environ.get("RESP_AI_PRELOAD_MODEL", "").lower() in {"1", "true", "yes"}:
    try:
        get_predictor()
    except Exception:
        # Keep process booting so /ready exposes the load failure to the caller.
        pass


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or "unknown.wav"
    extension = Path(filename).suffix.lower()
    if extension not in APP_CONFIG.allowed_extensions:
        supported = ", ".join(APP_CONFIG.allowed_extensions)
        return jsonify({"error": f"Unsupported file type: {extension}. Try one of: {supported}"}), 400

    try:
        payload = uploaded.read()
        if not payload:
            return jsonify({"error": "Empty upload received."}), 400

        suffix = extension or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)

        result = get_predictor().predict_path(temp_path, filename)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if any(
            hint in lowered
            for hint in (
                "format not recognised",
                "unknown format",
                "could not open",
                "error opening",
                "decode",
                "codec",
                "backend",
                "unsupported",
            )
        ):
            return jsonify(
                {
                    "error": (
                        "This recording could not be read. Try a clear WAV, MP3, M4A, AAC, OGG, FLAC, 3GP, or WEBM file."
                    )
                }
            ), 400
        return jsonify({"error": message}), 500
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host=APP_CONFIG.host, port=APP_CONFIG.port, debug=False)
