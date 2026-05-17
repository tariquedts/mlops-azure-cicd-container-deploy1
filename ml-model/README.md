# ML Model — YOLO Object Detection (Dockerized)

Two versions of a YOLOv8 object detection API, containerized and ready to deploy
via the Azure Container Apps CI/CD pipelines in this repo.

---

## Model Versions at a Glance

| Tag | Model | Parameters | mAP50-95 | Speed | Use Case |
|-----|-------|-----------|----------|-------|----------|
| `v1` | YOLOv8 Nano (`yolov8n`) | 6.3M | 37.3 | Fastest | Learning / quick demo |
| `v2` | YOLOv8 Small (`yolov8s`) | 22.5M | 44.9 | Fast | Better accuracy demo |

mAP scores are on the COCO dataset. Higher = more accurate.

---

## Folder Structure

```
ml-model/
├── v1/
│   ├── Dockerfile        # Builds image with YOLOv8 Nano weights baked in
│   ├── app.py            # FastAPI inference server
│   └── requirements.txt
├── v2/
│   ├── Dockerfile        # Builds image with YOLOv8 Small weights baked in
│   ├── app.py            # FastAPI inference server
│   └── requirements.txt
└── README.md
```

---

## Step 1 — Build and Push to Docker Hub

Replace `yourdockerhubuser` with your actual Docker Hub username.

```bash
# --- Build and push v1 (YOLOv8 Nano) ---
cd ml-model/v1
docker build -t yourdockerhubuser/yolo-mlops:v1 .
docker push yourdockerhubuser/yolo-mlops:v1

# --- Build and push v2 (YOLOv8 Small) ---
cd ../v2
docker build -t yourdockerhubuser/yolo-mlops:v2 .
docker push yourdockerhubuser/yolo-mlops:v2
```

> **Note:** Building downloads ~6MB (v1) or ~22MB (v2) of YOLOv8 weights
> from Ultralytics. The weights are baked into the image so the container
> starts instantly — no download at runtime.

---

## Step 2 — Test Locally Before Pushing to Azure

```bash
# Run v1 locally on port 8080
docker run -p 8080:80 yourdockerhubuser/yolo-mlops:v1

# Run v2 locally on port 8081
docker run -p 8081:80 yourdockerhubuser/yolo-mlops:v2
```

---

## Step 3 — Invoke the API

The container exposes 3 endpoints on port 80.

### Health Check
```bash
curl http://localhost:8080/health
```
Response:
```json
{ "status": "ok", "version": "v1", "model": "yolov8n" }
```

### Model Info
```bash
curl http://localhost:8080/model-info
```
Response:
```json
{
  "version": "v1",
  "model": "yolov8n",
  "description": "YOLOv8 Nano — lightweight model optimised for speed...",
  "num_classes": 80,
  "classes": { "0": "person", "1": "bicycle", ... }
}
```

### Predict — Send an Image
```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@/path/to/your/image.jpg"
```
Response:
```json
{
  "model_version": "v1",
  "model": "yolov8n",
  "image_filename": "image.jpg",
  "total_objects_detected": 3,
  "detections": [
    {
      "class": "person",
      "confidence": 0.9231,
      "bbox_xyxy": [120.5, 45.2, 380.1, 620.8]
    },
    {
      "class": "car",
      "confidence": 0.8754,
      "bbox_xyxy": [500.0, 200.0, 900.0, 550.0]
    }
  ]
}
```

`bbox_xyxy` = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]

### FastAPI Interactive Docs (Swagger UI)
```
http://localhost:8080/docs
```
Open this in a browser — you can upload images and test the API visually.

---

## Step 4 — Deploy via CI/CD Pipeline

Update the `IMAGE_TAG` variable in any pipeline file under `yml-pipelines/`:

```yaml
variables:
  IMAGE_TAG: 'v1'                               # <-- change to v2 and push
  DOCKER_IMAGE: 'yourdockerhubuser/yolo-mlops'
  CONTAINER_APP_NAME: 'yolo-ml-app'
  RESOURCE_GROUP: 'ml-rg'
  LOCATION: 'eastus'
  ENVIRONMENT_NAME: 'ml-env'
```

**To deploy v1:** set `IMAGE_TAG: 'v1'` → commit and push to master → pipeline runs → Azure Container App pulls `yourdockerhubuser/yolo-mlops:v1`

**To deploy v2:** set `IMAGE_TAG: 'v2'` → commit and push to master → pipeline runs → Azure Container App swaps to `yourdockerhubuser/yolo-mlops:v2`

---

## Step 5 — Invoke the Deployed Container on Azure

Once deployed, Azure Container Apps gives you a public URL like:
```
https://yolo-ml-app.{random}.eastus.azurecontainerapps.io
```

Use the same curl commands as local testing, replacing `localhost:8080` with your Azure URL:

```bash
# Health check on Azure
curl https://yolo-ml-app.{random}.eastus.azurecontainerapps.io/health

# Predict on Azure
curl -X POST https://yolo-ml-app.{random}.eastus.azurecontainerapps.io/predict \
  -F "file=@image.jpg"
```

---

## What Changes Between v1 and v2

| File | v1 | v2 |
|------|----|----|
| `app.py` | `MODEL_NAME = "yolov8n"` | `MODEL_NAME = "yolov8s"` |
| `Dockerfile` | Downloads `yolov8n.pt` | Downloads `yolov8s.pt` |
| Image size | ~700MB | ~750MB |
| Response `model` field | `"yolov8n"` | `"yolov8s"` |
| Detection accuracy | Lower | Higher |

The API contract (`/health`, `/model-info`, `/predict`) is identical in both versions —
only the underlying model weights differ. This is the key teaching point:
the CI/CD pipeline simply swaps the image tag, and students can observe
the difference in detection confidence scores between v1 and v2.
