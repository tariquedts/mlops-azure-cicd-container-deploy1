import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# v1 uses YOLOv8 Nano — smallest model, optimised for speed over accuracy
MODEL_NAME = "yolov8n"
MODEL_VERSION = "v1"
MODEL_DESCRIPTION = (
    "YOLOv8 Nano — lightweight model optimised for speed. "
    "6.3M parameters, mAP50-95: 37.3 on COCO dataset."
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
