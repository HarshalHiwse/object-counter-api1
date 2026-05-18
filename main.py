from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import io
import uuid
import os
from datetime import datetime

app = FastAPI(title="Mr.Count API", version="1.0")

model = YOLO("yolov8n.pt")

os.makedirs("temp", exist_ok=True)

@app.post("/count")
async def count_objects(
    file: UploadFile = File(...),
    class_name: str = Form(None),
    confidence: float = Form(0.4)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    results = model(image_np, conf=confidence, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    if class_name:
        class_name = class_name.lower()
        for k, v in model.names.items():
            if v.lower() == class_name:
                detections = detections[detections.class_id == k]
                break

    count = len(detections)

    # Annotate
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator()
    annotated = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = f"temp/{filename}"
    cv2.imwrite(output_path, annotated)

    return {
        "success": True,
        "count": count,
        "annotated_image_url": f"/temp/{filename}",
        "message": f"Found {count} objects"
    }

@app.get("/temp/{filename}")
async def get_image(filename: str):
    return FileResponse(f"temp/{filename}")

@app.get("/")
async def home():
    return {"message": "Mr.Count API is Running! Go to /docs"}
