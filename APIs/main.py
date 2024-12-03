from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from model import predict
from utils import read_image

app = FastAPI()

# define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# label mapping (for the 10 beer types)
label_mapping = {
    0: 'Tiger Beer',
    1: 'Asahi Super Dry Beer',
    2: 'Kingfisher Beer',
    3: 'Budweiser Beer',
    4: 'Stella Artois',
    5: 'Becks Beer',
    6: 'Carlsberg Beer',
    7: 'Heineken Beer',
    8: 'Tuborg Beer',
    9: 'Corona Beer'
}

@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    try:
        # read and preprocess the image
        image = await read_image(file)  
        image_tensor = transform(image).unsqueeze(0)  

        predicted_class = predict(image_tensor)
        predicted_beer = label_mapping.get(predicted_class, "Unknown Beer")
        
        return {"prediction": predicted_beer}
    except Exception as e:
        return {"error": str(e)}
