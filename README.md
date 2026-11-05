# UNO Q MyCard Vision

An Edge AI-powered business card detection service for **Arduino UNO Q**, using **QuadScan** and **NCNN** for on-device image processing.

This project is part of [UNO Q MyCard](https://github.com/Mighty-Net/uno-q-mycard_integrate) and provides a lightweight vision API for detecting business cards and identifying their four corners.

## Overview

MyCard Vision performs business card detection locally on the Arduino UNO Q without requiring cloud connectivity.

It uses a pretrained QuadScan model to identify document-like quadrilateral regions in camera images.

The service is designed to:

- Detect business cards in captured images.
- Identify the four corners of a detected card.
- Return detection results to MyCardOS.
- Run locally on UNO Q using CPU inference.
- Avoid additional model training.

## Architecture

```text
MyCardOS (React Web UI)
   |
   | Image
   v
Vision API (FastAPI)
   |
   v
Image Preprocessing
   |
   v
QuadScan (NCNN / CPU)
   |
   v
Corner Detection
   |
   v
Detection Result (JSON)
   |
   v
MyCardOS
```

## Components

| Component | Description |
|---|---|
| FastAPI | REST API for image detection |
| QuadScan | Pretrained quadrilateral detection model |
| NCNN | Neural network inference framework |
| CPU | On-device inference execution |
| MyCardOS | Frontend application consuming detection results |

## Detection Workflow

1. MyCardOS captures an image from the camera stream.
2. The image is sent to the Vision API.
3. The image is preprocessed for model inference.
4. QuadScan detects the business card region.
5. The four corner coordinates are extracted.
6. The detection result is returned to MyCardOS.

All detection processing is performed locally on the UNO Q.

Cloud AI is used separately for business card information recognition.

## API

The Vision service exposes an HTTP API for business card detection.

| Property | Value |
|---|---|
| Framework | FastAPI |
| Port | 18081 |
| Input | Camera image |
| Output | Detection result and corner coordinates |

The service is intended to be accessed by MyCardOS within the UNO Q application.

## Performance

The Vision service is designed for lightweight edge inference.

- Uses a pretrained model without additional training.
- Executes inference locally on the CPU.
- Uses NCNN for efficient neural network execution.
- Reduces image resolution during preprocessing to improve performance.
- Avoids unnecessary cloud requests during card detection.

Actual inference speed depends on image resolution, CPU load, and preprocessing configuration.

## Notes

- Business card detection runs entirely on UNO Q.
- QuadScan is used for quadrilateral detection rather than OCR.
- Cloud AI (Gemma 4) is responsible for extracting business card information.
- Detection accuracy depends on lighting, camera focus, image quality, and card positioning.
- CPU inference is used for the current implementation.
- No custom model training is required for the current demo.

## License

See [LICENSE](LICENSE) for licensing information.