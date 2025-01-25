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
from fuzzywuzzy import process
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


with open('C:/Users/suj33/Desktop/iBeer.ai/beerInformationRetrival/infoRetrivalAPI/beerInformatonDataset/data.json') as f:
    beers_data = json.load(f)


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
classification_model.load_state_dict(torch.load('C:/Users/suj33/Desktop/iBeer.ai/beerLabelClassification/weights/label_classifiations.pth', map_location=device))
classification_model.eval()

# Transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

detection_model = YOLO("C:/Users/suj33/Desktop/iBeer.ai/beerLabelDetector/custom_model/weights/best.pt")

# Utility function to classify beer
def classify_beer(cropped_img):
    image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return reverse_label_mapping[predicted_class.item()]

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

# Load beer recommendation data
file_path = 'C:/Users/suj33/Desktop/iBeer.ai/beerRecommendation/beerRecSysAPI/beerDataset/cleaned_beer_dataset.csv'
dataframe = pd.read_csv(file_path)

dataframe['ingredients'] = dataframe['ingredients'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')
dataframe['flavor_profile'] = dataframe['flavor_profile'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')
dataframe['pairings'] = dataframe['pairings'].apply(lambda x: ', '.join(x.split(', ')) if isinstance(x, str) else '')

dataframe['combined_features'] = (
    dataframe['brand'] + " " +
    dataframe['style'] + " " +
    dataframe['ingredients'] + " " +
    dataframe['flavor_profile'] + " " +
    dataframe['pairings'] + " " +
    dataframe['description']
)

vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(dataframe['combined_features'])

svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(feature_matrix)

cosine_sim = cosine_similarity(latent_matrix, latent_matrix)

def get_beer_recommendation(beer_name: str):
    if beer_name not in dataframe['beer_name'].values:
        suggested_name = process.extractOne(beer_name, dataframe['beer_name'].unique())[0]
        beer_name = suggested_name

    idx = dataframe[dataframe['beer_name'] == beer_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    beer_indices = [i[0] for i in sim_scores if i[0] != idx] 
    recommended_beers = dataframe['beer_name'].iloc[beer_indices].drop_duplicates().head(3).tolist()

    recommendations = []
    for beer in recommended_beers:
        beer_info = get_beer_info(beer)
        recommendations.append(beer_info)

    return recommendations

# Endpoints
@app.post("/detect/")
async def detect_labels(file: UploadFile = File(...)):
    try:
        image_path = file.filename
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        results = detection_model.predict(image_path)
        os.remove(image_path)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class": results[0].names[int(box.cls[0])]
            })

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        image_path = file.filename
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        img = cv2.imread(image_path)
        os.remove(image_path)

        results = detection_model.predict(img)
        result = results[0]

        predictions = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            predictions.append(classify_beer(cropped_img))

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/beer_info/{beer_name}")
def get_info(beer_name: str):
    beer_info = get_beer_info(beer_name)
    if "error" in beer_info:
        return JSONResponse(status_code=404, content=beer_info)
    return beer_info

@app.get("/beer-recommendation")
async def get_beer_recommendation_endpoint(beer_name: str):
    recommendations = get_beer_recommendation(beer_name)
    return {"recommended_beers": recommendations}
