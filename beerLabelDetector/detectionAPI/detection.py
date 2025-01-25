from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

model = YOLO("C:/Users/suj33/Desktop/iBeer.ai/beerLabelDetector/custom_model/weights/best.pt")

DETECTED_LABELS_PATH = "C:/Users/suj33/Desktop/iBeer.ai/beerLabelDetector/detectionAPI/detectedLabels"
os.makedirs(DETECTED_LABELS_PATH, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Detect labels in the uploaded image using YOLO and save the result.
    """

    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    results = model.predict(img)
    result = results[0]
    boxes = result.boxes
    classes = result.names


    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])


        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img, 
            f"{classes[class_id]} {confidence:.2f}", 
            (int(x1), int(y1) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )


    unique_filename = f"detected_{uuid.uuid4().hex}.jpeg"
    annotated_image_path = os.path.join(DETECTED_LABELS_PATH, unique_filename)
    cv2.imwrite(annotated_image_path, img)

    return JSONResponse(content={
        "message": "Label detected successfully!",
        "annotated_image_path": annotated_image_path
    })
