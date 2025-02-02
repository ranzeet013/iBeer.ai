from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from collections import OrderedDict
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

app = FastAPI()

# Load beers data
with open('C:/Users/suj33/Desktop/New folder (2)/testing/data.json') as f:
    beers_data = json.load(f)

# Load recommendation data
file_path = 'C:/Users/suj33/Desktop/New folder (2)/testing/cleaned_beer_dataset.csv'
dataframe = pd.read_csv(file_path)

dataframe['combined_features'] = (
    dataframe['brand'] + " " +
    dataframe['style'] + " " +
    dataframe['ingredients'].fillna('') + " " +
    dataframe['flavor_profile'].fillna('') + " " +
    dataframe['pairings'].fillna('') + " " +
    dataframe['description'].fillna('')
)

vectorizer = CountVectorizer(stop_words='english', binary=True)
feature_matrix = vectorizer.fit_transform(dataframe['combined_features'])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Label mapping for classification
label_mapping = { 'Tiger Beer': 0, 'Asahi Super Dry Beer': 1, 'Kingfisher Beer': 2, 'Budweiser Beer': 3,
    'Stella Artois': 4, 'Becks Beer': 5, 'Carlsberg Beer': 6, 'Heineken Beer': 7, 'Tuborg Beer': 8, 'Corona Beer': 9 }
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define classification model
class BeerModelInference(nn.Module):
    def __init__(self):
        super(BeerModelInference, self).__init__()
        self.network = models.vgg16(pretrained=True)
        self.network.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 512)), ('relu', nn.ReLU()), ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 10)), ('output', nn.LogSoftmax(dim=1))
        ]))
    def forward(self, xb): return self.network(xb)

# Load classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = BeerModelInference().to(device)
classification_model.load_state_dict(torch.load('C:/Users/suj33/Desktop/New folder (2)/testing/label_classifiations.pth', map_location=device))
classification_model.eval()

# YOLO detection model
detection_model = YOLO("C:/Users/suj33/Desktop/New folder (2)/testing/best.pt")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classify beer
def classify_beer(cropped_img):
    image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classification_model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return reverse_label_mapping[predicted_class.item()]

# Beer info retrieval
def get_best_match(input_name, beers_list):
    names = [beer['name_of_beer'] for beer in beers_list]
    return process.extractOne(input_name, names)

def get_beer_info(beer_name):
    best_match = get_best_match(beer_name, beers_data)
    if best_match and best_match[1] > 80:
        for beer in beers_data:
            if beer['name_of_beer'] == best_match[0]:
                return {key: beer[key] for key in beer.keys()}
    return {"error": "No similar beer found."}

# Beer recommendation
def get_recommendations(beer_name):
    if beer_name not in dataframe['beer_name'].values:
        suggested_name = process.extractOne(beer_name, dataframe['beer_name'].unique())[0]
        beer_name = suggested_name

    idx = dataframe[dataframe['beer_name'] == beer_name].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    beer_indices = [i[0] for i in sim_scores if i[0] != idx]
    return dataframe['beer_name'].iloc[beer_indices].drop_duplicates().head(3).tolist()

# Endpoints
@app.post("/detect/")
async def detect_labels(file: UploadFile = File(...)):
    try:
        image_path = file.filename
        with open(image_path, "wb") as buffer: buffer.write(await file.read())
        results = detection_model.predict(image_path)
        os.remove(image_path)

        detections = [{"box": list(map(int, box.xyxy[0])), "confidence": float(box.conf[0]), "class": results[0].names[int(box.cls[0])]} for box in results[0].boxes]
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        image_path = file.filename
        with open(image_path, "wb") as buffer: buffer.write(await file.read())
        img = cv2.imread(image_path)
        os.remove(image_path)

        results = detection_model.predict(img)
        predictions = [classify_beer(img[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]) for box in results[0].boxes]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/beer_info/{beer_name}")
def get_info(beer_name: str):
    beer_info = get_beer_info(beer_name)
    return JSONResponse(status_code=404, content=beer_info) if "error" in beer_info else beer_info

@app.get("/recommend/{beer_name}")
def recommend(beer_name: str):
    recommendations = get_recommendations(beer_name)
    return {"Recommended Beers": recommendations}
