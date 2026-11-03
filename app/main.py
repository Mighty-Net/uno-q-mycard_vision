import io
import os
import time
import cv2
from collections import deque

import ncnn
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

MODEL_BASE = "/app/models/lcnet100_bifpn_4d.ncnn"
MAX_IMAGE_SIZE = 5 * 1024 * 1024
HEATMAP_THRESHOLD = 0.2

# ncnn CPU FP32; keep the existing ONNX Runtime API untouched.
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 2
net.opt.use_fp16_storage = False
net.opt.use_fp16_arithmetic = False
net.opt.use_fp16_packed = False

if net.load_param(MODEL_BASE + ".param") != 0:
    raise RuntimeError("Failed to load ncnn model parameters")
if net.load_model(MODEL_BASE + ".bin") != 0:
    raise RuntimeError("Failed to load ncnn model weights")

INPUT_NAME = "in0"
OUTPUT_NAME = "out0"


def preprocess(image: Image.Image):
    # Pillow RGB → NumPy
    img = np.asarray(image.convert("RGB"))

    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize
    img = cv2.resize(
        img,
        (256, 256),
        interpolation=cv2.INTER_LINEAR,
    )

    # 反相
    img = cv2.bitwise_not(img)

    # Normalize + NCHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img *= 1.0 / 255.0

    return img[None]


def largest_component_center(mask: np.ndarray):
    binary = np.ascontiguousarray(mask, dtype=np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1

    return centroids[idx].astype(np.float32)


def decode_heatmap(
    heatmap: np.ndarray,
    image_width: int,
    image_height: int,
):
    """Decode a [1,4,128,128] heatmap into TL, TR, BR, BL corners."""
    if heatmap.shape != (1, 4, 128, 128):
        raise ValueError(f"Unexpected heatmap shape: {heatmap.shape}")

    points = []

    for channel in heatmap[0]:
        mask = channel >= HEATMAP_THRESHOLD
        center = largest_component_center(mask)

        if center is None:
            return None

        x, y = center
        x = float((x + 0.5) * image_width / 128)
        y = float((y + 0.5) * image_height / 128)

        points.append({
            "x": max(0.0, min(x, image_width - 1.0)),
            "y": max(0.0, min(y, image_height - 1.0)),
        })

    return {
        "topLeft": points[0],
        "topRight": points[1],
        "bottomRight": points[2],
        "bottomLeft": points[3],
    }


def infer(tensor: np.ndarray) -> np.ndarray:
    """Run ncnn CPU and return ONNX-compatible [1,4,128,128] heatmap."""
    mat = ncnn.Mat(tensor)
    extractor = net.create_extractor()

    ret = extractor.input(INPUT_NAME, mat)
    if ret != 0:
        raise RuntimeError(f"ncnn input failed: {ret}")

    ret, output = extractor.extract(OUTPUT_NAME)
    if ret != 0:
        raise RuntimeError(f"ncnn inference failed: {ret}")

    heatmap = np.asarray(output).copy()
    if heatmap.shape != (4, 128, 128):
        raise RuntimeError(f"Unexpected ncnn output shape: {heatmap.shape}")
    if not np.isfinite(heatmap).all():
        raise RuntimeError("ncnn output contains NaN or Infinity")

    return heatmap[None]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_BASE),
        "providers": ["ncnn CPU FP32"],
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    total_start = time.perf_counter()

    data = await file.read(MAX_IMAGE_SIZE + 1)

    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5MB")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image")

    width, height = image.size

    start = time.perf_counter()
    tensor = preprocess(image)
    preprocess_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    heatmap = infer(tensor)
    inference_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    corners = decode_heatmap(heatmap, width, height)
    postprocess_ms = (time.perf_counter() - start) * 1000

    return {
        "success": corners is not None,
        "corners": corners,
        "width": width,
        "height": height,
        "preprocess_ms": round(preprocess_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "postprocess_ms": round(postprocess_ms, 1),
        "total_ms": round((time.perf_counter() - total_start) * 1000, 1),
    }
