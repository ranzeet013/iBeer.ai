from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from collections import OrderedDict
from fuzzywuzzy import process
import json
import numpy as np
from io import BytesIO
import pandas as pd

# Load beers data
with open('C:/Users/suj33/Desktop/New folder (2)/testing/data.json') as f:
    beers_data = json.load(f)

# Label mapping for classification
label_mapping = {
    'Tiger Beer': 0,
    'Asahi Super Dry Beer': 1,
    'Kingfisher Beer': 2,
    'Budweiser Beer': 3,
    'Stella Artois': 4,
    'Becks Beer': 5,
    'Carlsberg Beer': 6,
    'Heineken Beer': 7,
    'Tuborg Beer': 8,
    'Corona Beer': 9
}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define the classification model
class BeerModelInference(nn.Module):
    def __init__(self):
        super(BeerModelInference, self).__init__()
        self.network = models.vgg16(pretrained=True)
        self.network.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 512)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 10)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, xb):
        return self.network(xb)

# Load the classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = BeerModelInference().to(device)
classification_model.load_state_dict(torch.load('C:/Users/suj33/Desktop/New folder (2)/testing/label_classifiations.pth', map_location=device))
classification_model.eval()

# Transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classification prediction function
def classify_beer(cropped_img):
    image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

    return reverse_label_mapping[predicted_class]

# Retrieve beer information
def get_best_match(input_name: str, beers_list: list):
    names = [beer['name_of_beer'] for beer in beers_list]
    best_match = process.extractOne(input_name, names)
    return best_match

def get_beer_info(beer_name: str):
    best_match = get_best_match(beer_name, beers_data)
    if best_match and best_match[1] > 80:
        matched_beer_name = best_match[0]
        for beer in beers_data:
            if beer['name_of_beer'] == matched_beer_name:
                return {
                    "name_of_beer": beer["name_of_beer"],
                    "history_and_background": beer["history_and_background"],
                    "ingredients": beer["ingredients"],
                    "brewing_process": beer["brewing_process"],
                    "note": beer["note"],
                    "food_pairing": beer["food_pairing"],
                    "smokes": beer["smokes"]
                }
    else:
        return {"error": "No similar beer found. Please check the name."}

# Load the YOLO detection model
detection_model = YOLO("C:/Users/suj33/Desktop/New folder (2)/testing/best.pt")

# Initialize FastAPI app
app = FastAPI()

@app.post("/process-beer/")
async def process_beer(image: UploadFile = File(...)):
    try:
        # Read the input image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform detection
        results = detection_model.predict(img)
        result = results[0]

        # Get detection details
        boxes = result.boxes

        if not boxes:
            return JSONResponse(content={"message": "No beer detected in the image."}, status_code=200)

        predictions = []

        # Process each detected box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            # Crop the detected region
            cropped_img = img[y1:y2, x1:x2]

            # Classify the cropped region
            predicted_label = classify_beer(cropped_img)
            beer_info = get_beer_info(predicted_label)

            predictions.append({
                "detecting_label": "Label detected successfully",
                "detected_beer": predicted_label,
                "confidence": float(confidence),
                "information": beer_info if "error" not in beer_info else None,
                "error": beer_info.get("error") if "error" in beer_info else None
            })

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
