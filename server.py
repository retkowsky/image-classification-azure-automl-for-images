"""
FastAPI server for ONNX image classification inference.

Loads an ONNX model at startup and exposes a /predict endpoint
that accepts an image file and returns top-K class predictions.

Usage:
    pip install fastapi uvicorn onnxruntime pillow numpy python-multipart
    python server.py --model model.onnx --labels labels.txt --port 8000

Then POST an image:
    curl -X POST http://localhost:8000/predict \
         -F "file=@test_image.jpg" | python -m json.tool
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn


# ─────────────── Globals ───────────────

app = FastAPI(title="ONNX Classification Server", version="1.0.0")
logger = logging.getLogger("onnx-server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None
labels: list[str] = []
input_name: str = ""
output_name: str = ""
input_height: int = 224
input_width: int = 224
norm_mean: list[float] = [0.485, 0.456, 0.406]
norm_std: list[float] = [0.229, 0.224, 0.225]


# ─────────────── Preprocessing ───────────────


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and convert an image to a NCHW float32 tensor.

    Args:
        image: PIL Image in RGB mode.

    Returns:
        np.ndarray of shape [1, 3, H, W] with float32 dtype.
    """
    image = image.convert("RGB").resize(
        (input_width, input_height), Image.BILINEAR
    )
    arr = np.array(image, dtype=np.float32) / 255.0

    arr = (arr - np.array(norm_mean, dtype=np.float32)) / np.array(
        norm_std, dtype=np.float32
    )

    # HWC -> NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax over a 1D array.

    Args:
        logits: Raw model output logits.

    Returns:
        Probability distribution that sums to 1.0.
    """
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


# ─────────────── Routes ───────────────


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint to verify the server and model are ready."""
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "num_labels": len(labels),
        "input_shape": [1, 3, input_height, input_width],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top predictions"),
    apply_softmax: bool = Query(default=True, description="Apply softmax to logits"),
) -> dict:
    """Run inference on an uploaded image and return top-K predictions.

    Args:
        file: Uploaded image file.
        top_k: Number of top predictions to return.
        apply_softmax: Whether to apply softmax to raw logits.

    Returns:
        dict with predictions list and inference timing.
    """
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Preprocess
    tensor = preprocess_image(image)

    # Inference
    t0 = time.perf_counter()
    outputs = session.run([output_name], {input_name: tensor})
    inference_ms = (time.perf_counter() - t0) * 1000

    scores = outputs[0][0]
    if apply_softmax:
        scores = softmax(scores)

    # Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        label = labels[int(idx)] if int(idx) < len(labels) else f"class_{idx}"
        predictions.append(
            {
                "label": label,
                "score": round(float(scores[idx]), 6),
                "index": int(idx),
            }
        )

    return {
        "predictions": predictions,
        "inference_ms": round(inference_ms, 2),
    }


# ─────────────── Startup ───────────────


def load_model(model_path: str, labels_path: str | None = None) -> None:
    """Load the ONNX model and optional labels file into global state.

    Args:
        model_path: Path to the .onnx model file.
        labels_path: Path to a text file with one label per line.
    """
    global session, labels, input_name, output_name, input_height, input_width

    logger.info(f"Loading model: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=ort.get_available_providers(),
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name

    # Try to extract H, W from input shape (NCHW assumed)
    shape = input_meta.shape
    if len(shape) == 4 and all(isinstance(d, int) for d in shape):
        input_height = shape[2]
        input_width = shape[3]

    logger.info(f"  Input:  {input_name} -> {shape}")
    logger.info(f"  Output: {output_name} -> {output_meta.shape}")
    logger.info(f"  Resolved input size: {input_width}x{input_height}")
    logger.info(f"  Providers: {ort.get_available_providers()}")

    if labels_path and Path(labels_path).exists():
        text = Path(labels_path).read_text()
        labels.clear()

        # Try JSON first (array or object)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                labels.extend(str(item) for item in parsed)
            elif isinstance(parsed, dict):
                labels.extend(str(v) for v in parsed.values())
            else:
                raise ValueError("Not a list or dict")
        except (json.JSONDecodeError, ValueError):
            # Fallback: one label per line
            labels.extend(line.strip() for line in text.splitlines() if line.strip())

        logger.info(f"  Labels: {len(labels)} classes loaded")
        logger.info(f"  First 5: {labels[:5]}")
    else:
        logger.warning("  No labels file — predictions will use class indices")


# ─────────────── CLI ───────────────


def main() -> None:
    """Parse CLI arguments and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="ONNX Classification Server")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the .onnx model file"
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="Path to labels .txt file"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean R G B (default: ImageNet)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std R G B (default: ImageNet)",
    )
    args = parser.parse_args()

    global norm_mean, norm_std
    norm_mean = args.mean
    norm_std = args.std

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    load_model(args.model, args.labels)

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
