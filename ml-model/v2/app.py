import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# v2 uses YOLOv8 Small — better accuracy than Nano, still efficient
MODEL_NAME = "yolov8s"
MODEL_VERSION = "v2"
MODEL_DESCRIPTION = (
    "YOLOv8 Small — improved accuracy over Nano. "
    "22.5M parameters, mAP50-95: 44.9 on COCO dataset."
)

app = FastAPI(
    title="YOLO Object Detection API",
    description=MODEL_DESCRIPTION,
    version=MODEL_VERSION,
)

model = YOLO(f"{MODEL_NAME}.pt")


@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION, "model": MODEL_NAME}


@app.get("/model-info")
def model_info():
    return {
        "version": MODEL_VERSION,
        "model": MODEL_NAME,
        "description": MODEL_DESCRIPTION,
        "num_classes": len(model.names),
        "classes": model.names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
            })

    return JSONResponse({
        "model_version": MODEL_VERSION,
        "model": MODEL_NAME,
        "image_filename": file.filename,
        "total_objects_detected": len(detections),
        "detections": detections,
    })
