from __future__ import annotations

import base64
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from resp_ai.config import load_audio_config, load_inference_config, load_yaml
from resp_ai.features.audio import extract_window_batch_from_path
from resp_ai.labels import CLASS_NAMES
from resp_ai.paths import project_root_from_config, resolve_project_path

LEGACY_KERAS_MODULE_MAP = {
    "keras.src.engine.functional": "keras.src.models.functional",
}


def _strip_null_quantization_config(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "quantization_config" and item is None:
                continue
            cleaned[key] = _strip_null_quantization_config(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_null_quantization_config(item) for item in value]
    return value


def _rewrite_legacy_keras_modules(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "module" and isinstance(item, str):
                cleaned[key] = LEGACY_KERAS_MODULE_MAP.get(item, item)
            else:
                cleaned[key] = _rewrite_legacy_keras_modules(item)
        return cleaned
    if isinstance(value, list):
        return [_rewrite_legacy_keras_modules(item) for item in value]
    return value


def _normalize_layer_config_shapes(value):
    if isinstance(value, dict):
        cleaned = {}
        class_name = value.get("class_name")
        for key, item in value.items():
            cleaned[key] = _normalize_layer_config_shapes(item)

        layer_config = cleaned.get("config")
        if (
            class_name in {"BatchNormalization", "Normalization"}
            and isinstance(layer_config, dict)
            and isinstance(layer_config.get("axis"), list)
            and len(layer_config["axis"]) == 1
        ):
            layer_config["axis"] = layer_config["axis"][0]
        if (
            class_name == "DepthwiseConv2D"
            and isinstance(layer_config, dict)
            and layer_config.get("groups") == 1
        ):
            layer_config.pop("groups", None)
        return cleaned
    if isinstance(value, list):
        return [_normalize_layer_config_shapes(item) for item in value]
    return value


def load_model_with_compat(model_path: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        message = str(exc)
        if "quantization_config" not in message and "keras.src.engine.functional" not in message:
            raise

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        extracted_dir = temp_root / "model"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(model_path, "r") as archive:
            archive.extractall(extracted_dir)

        config_path = extracted_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cleaned_config = _strip_null_quantization_config(config)
        cleaned_config = _rewrite_legacy_keras_modules(cleaned_config)
        cleaned_config = _normalize_layer_config_shapes(cleaned_config)
        config_path.write_text(json.dumps(cleaned_config), encoding="utf-8")

        sanitized_model_path = temp_root / "sanitized_model.keras"
        with zipfile.ZipFile(sanitized_model_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in extracted_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(extracted_dir))

        return tf.keras.models.load_model(sanitized_model_path, compile=False)


class Predictor:
    def __init__(self, config_path: str | Path, model_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.project_root = project_root_from_config(self.config_path)
        self.config = load_yaml(self.config_path)
        self.audio_config = load_audio_config(self.config)
        self.inference_config = load_inference_config(self.config)
        self.enable_gradcam = os.environ.get("RESP_AI_ENABLE_GRADCAM", "").lower() in {"1", "true", "yes"}

        if model_path is None:
            model_path = resolve_project_path(self.project_root, self.config["paths"]["models_root"]) / "latest" / "best_model.keras"
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = load_model_with_compat(self.model_path)
        self.class_names = self._load_class_names()
        self.last_conv_layer_name = self._find_last_conv_layer()
        self.grad_model = None
        if self.enable_gradcam:
            self.grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(self.last_conv_layer_name).output,
                    self._prediction_output_tensor(),
                ],
            )

    def _load_class_names(self) -> list[str]:
        class_names_path = self.model_path.parent / "class_names.json"
        if class_names_path.exists():
            with class_names_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return CLASS_NAMES

    def _find_last_conv_layer(self) -> str:
        for preferred_name in ("gradcam_conv",):
            try:
                self.model.get_layer(preferred_name)
                return preferred_name
            except ValueError:
                pass

        for layer in reversed(self.model.layers):
            output_tensor = getattr(layer, "output", None)
            if output_tensor is None:
                continue
            output_shape = getattr(output_tensor, "shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name
        raise ValueError("No convolutional layer found for Grad-CAM.")

    def _prediction_output_tensor(self):
        outputs = self.model.outputs if isinstance(self.model.outputs, (list, tuple)) else [self.model.output]
        if not outputs:
            raise ValueError("Model does not expose an output tensor for Grad-CAM.")
        return outputs[0]

    def predict_path(self, audio_path: str | Path, filename: str) -> dict:
        batch, window_metadata = extract_window_batch_from_path(
            str(audio_path),
            self.audio_config,
            overlap=self.inference_config.window_overlap,
            max_windows=self.inference_config.max_windows,
        )
        window_probs = self.model.predict(batch, verbose=0)
        probs = self._aggregate_window_probabilities(window_probs)
        probs = self._apply_pneumonia_postprocess(probs, window_probs)
        pred_index = int(np.argmax(probs))
        prediction = self.class_names[pred_index]
        confidence = float(probs[pred_index] * 100.0)
        representative_window_index = int(np.argmax(window_probs[:, pred_index]))
        representative_batch = batch[representative_window_index : representative_window_index + 1]
        sorted_indices = np.argsort(probs)[::-1]
        secondary_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else pred_index
        heatmap = ""

        windows = []
        for metadata, prob_row in zip(window_metadata, window_probs):
            window_prediction_index = int(np.argmax(prob_row))
            windows.append(
                {
                    "window_index": int(metadata["window_index"]),
                    "start_sec": float(metadata["start_sec"]),
                    "duration_sec": float(metadata["duration_sec"]),
                    "prediction": self.class_names[window_prediction_index],
                    "confidence": round(float(prob_row[window_prediction_index] * 100.0), 2),
                    }
                )

        try:
            heatmap = self._gradcam_data_url(representative_batch, pred_index)
        except Exception:
            heatmap = ""

        return {
            "filename": filename,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                class_name: round(float(prob * 100.0), 2)
                for class_name, prob in zip(self.class_names, probs)
            },
            "secondary_prediction": self.class_names[secondary_index],
            "secondary_confidence": round(float(probs[secondary_index] * 100.0), 2),
            "severity": self._severity(prediction, confidence),
            "heatmap": heatmap,
            "windows_used": len(window_metadata),
            "aggregation": self.inference_config.aggregation,
            "window_overlap": self.inference_config.window_overlap,
            "representative_window": windows[representative_window_index],
            "window_predictions": windows,
        }

    def _aggregate_window_probabilities(self, window_probs: np.ndarray) -> np.ndarray:
        mean_probs = window_probs.mean(axis=0)
        strategy = self.inference_config.aggregation
        if strategy != "pneumonia_sensitive":
            return mean_probs

        probs = mean_probs.copy()
        max_probs = window_probs.max(axis=0)

        if "Pneumonia" in self.class_names:
            pneumonia_index = self.class_names.index("Pneumonia")
            probs[pneumonia_index] = (0.4 * mean_probs[pneumonia_index]) + (0.6 * max_probs[pneumonia_index])

        if "COPD" in self.class_names:
            copd_index = self.class_names.index("COPD")
            probs[copd_index] = (0.9 * mean_probs[copd_index]) + (0.1 * max_probs[copd_index])

        total = float(np.sum(probs))
        if total > 0:
            probs = probs / total
        return probs

    def _apply_pneumonia_postprocess(self, probs: np.ndarray, window_probs: np.ndarray) -> np.ndarray:
        if "Pneumonia" not in self.class_names or "COPD" not in self.class_names:
            return probs

        pneumonia_index = self.class_names.index("Pneumonia")
        copd_index = self.class_names.index("COPD")
        current_winner = int(np.argmax(probs))
        if current_winner != copd_index:
            return probs

        pneumonia_probs = window_probs[:, pneumonia_index]
        pneumonia_peak = float(np.max(pneumonia_probs))
        pneumonia_mean = float(probs[pneumonia_index])
        copd_mean = float(probs[copd_index])
        pneumonia_window_share = float(np.mean(np.argmax(window_probs, axis=1) == pneumonia_index))

        should_promote = (
            pneumonia_peak >= 0.60
            and pneumonia_window_share >= 0.20
            and pneumonia_mean >= 0.18
            and (copd_mean - pneumonia_mean) <= 0.20
        )
        if not should_promote:
            return probs

        adjusted = probs.copy()
        boost = min(0.08, max(0.0, copd_mean - pneumonia_mean + 0.01))
        adjusted[pneumonia_index] += boost
        adjusted[copd_index] = max(0.0, adjusted[copd_index] - boost)
        total = float(np.sum(adjusted))
        if total > 0:
            adjusted = adjusted / total
        return adjusted

    def _severity(self, prediction: str, confidence: float) -> dict:
        if confidence <= 50:
            level = "Low"
        elif confidence <= 80:
            level = "Moderate"
        else:
            level = "High"

        message_map = {
            "Normal": {
                "Low": "Breathing appears normal, but confidence is low. Try recording again in a quiet environment.",
                "Moderate": "No abnormal lung patterns detected. Maintain healthy habits and avoid pollution or smoking exposure.",
                "High": "Breathing appears normal with high confidence. No immediate medical action is required.",
            },
            "Asthma": {
                "Low": "Possible mild asthma symptoms detected. Re-record if needed and monitor symptoms carefully.",
                "Moderate": "Signs of asthma detected. Use prescribed inhalers if available and consult a doctor if symptoms persist.",
                "High": "Strong indication of asthma. Immediate medical consultation is recommended.",
            },
            "COPD": {
                "Low": "Early signs of COPD may be present. Avoid smoking and monitor your breathing regularly.",
                "Moderate": "COPD symptoms detected. Seek medical advice for proper diagnosis and follow prescribed care.",
                "High": "Severe COPD symptoms detected. Immediate medical attention is recommended.",
            },
            "Pneumonia": {
                "Low": "Possible early signs of infection detected. Monitor symptoms such as cough or fever.",
                "Moderate": "Likely pneumonia detected. Consult a doctor promptly and consider further medical tests.",
                "High": "Strong indication of pneumonia. Urgent medical treatment may be required.",
            },
        }

        disease_messages = message_map.get(prediction)
        if disease_messages is not None:
            return {"level": level, "message": disease_messages[level]}

        return {
            "level": level,
            "message": f"{prediction} pattern detected with {level.lower()} confidence. Clinical review is recommended.",
        }

    def _gradcam_data_url(self, batch: np.ndarray, class_index: int) -> str:
        if not self.enable_gradcam or self.grad_model is None:
            return self._lightweight_heatmap_data_url(batch)

        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(batch, training=False)
            loss = predictions[:, class_index]

        gradients = tape.gradient(loss, conv_output)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)
        base_image = np.clip(batch[0][:, :, 0].astype(np.float32), 0.0, 1.0)
        heatmap = tf.image.resize(
            heatmap[tf.newaxis, ..., tf.newaxis],
            size=base_image.shape,
            method="bilinear",
        )[0, :, :, 0].numpy()
        heatmap = np.clip(heatmap, 0.0, 1.0)

        base_rgb = np.repeat(base_image[..., np.newaxis], 3, axis=-1)
        accent = np.stack(
            [
                heatmap,
                np.sqrt(heatmap),
                np.power(heatmap, 3),
            ],
            axis=-1,
        )
        alpha = (heatmap[..., np.newaxis] * 0.55).astype(np.float32)
        blended = np.clip((base_rgb * (1.0 - alpha)) + (accent * alpha), 0.0, 1.0)
        png_bytes = tf.io.encode_png((blended * 255.0).astype(np.uint8)).numpy()
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _lightweight_heatmap_data_url(self, batch: np.ndarray) -> str:
        base_image = np.clip(batch[0][:, :, 0].astype(np.float32), 0.0, 1.0)
        time_energy = base_image.mean(axis=0)
        time_energy = (time_energy - time_energy.min()) / (time_energy.max() - time_energy.min() + 1e-6)
        focus_map = np.repeat(time_energy[np.newaxis, :], base_image.shape[0], axis=0)

        base_rgb = np.repeat(base_image[..., np.newaxis], 3, axis=-1)
        accent = np.stack(
            [
                focus_map * 0.25,
                focus_map * 0.7,
                focus_map,
            ],
            axis=-1,
        )
        alpha = (focus_map[..., np.newaxis] * 0.42).astype(np.float32)
        blended = np.clip((base_rgb * (1.0 - alpha)) + (accent * alpha), 0.0, 1.0)
        png_bytes = tf.io.encode_png((blended * 255.0).astype(np.uint8)).numpy()
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
