from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fuzzywuzzy import process
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

# Initialize FastAPI app
app = FastAPI()

# CORS settings to allow API access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Paths and data
BASE_DIR = Path("C:/Users/suj33/Desktop/iBeer.ai/poster/realImage")

# Load the beer data
with open('C:/Users/suj33/Desktop/iBeer.ai/infoRetrival/data.json') as f:
    beers_data = json.load(f)

# Beer model (this will match the structure of your beer data)
class Beer(BaseModel):
    name_of_beer: str
    history_and_background: str
    ingredients: dict
    brewing_process: dict
    similar_beer: list
    food_pairing: str

# Utility function to perform fuzzy search on beer names
def get_best_match(input_name: str, beers_list: list):
    # Get the best matching beer name using fuzzywuzzy's process.extractOne
    names = [beer['name_of_beer'] for beer in beers_list]
    best_match = process.extractOne(input_name, names)
    return best_match

# Model configuration
MODEL_PATH = "C:/Users/suj33/Desktop/iBeer.ai/ClassificationAPI/model/beer_classifiations.pth"
DEVICE = "cpu"

# BeerModel class definition (same as in model.py)
class BeerModel(nn.Module):
    def __init__(self):
        super(BeerModel, self).__init__()
        # Using a pre-trained VGG16 model for transfer learning
        self.network = models.vgg16(pretrained=True)

        # Freeze all layers except the final classifier
        for param in self.network.parameters():
            param.requires_grad = False

        # Update the classifier to have 10 output classes (for 10 beers)
        self.network.classifier[6] = nn.Linear(self.network.classifier[6].in_features, 10)

    def forward(self, x):
        return self.network(x)

# Load model function
model = None
def load_model():
    global model
    if model is None:
        model = BeerModel()
        try:
            # Load the model weights
            state_dict = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    return model

# Prediction function
def predict(image_tensor):
    model = load_model()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Label mapping for the 10 beer types
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

# Image reading function
async def read_image(file: UploadFile) -> Image.Image:
    """Read and return the uploaded image file."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

# Routes for Beer Info API
@app.get("/beer/{beer_name}", response_model=Beer)
async def get_beer_info(beer_name: str):
    # Perform fuzzy search to find the best matching beer name
    best_match = get_best_match(beer_name, beers_data)
    if best_match and best_match[1] > 80:  # If confidence is greater than 80%
        matched_beer_name = best_match[0]
        # Find the beer data corresponding to the matched name
        for beer in beers_data:
            if beer['name_of_beer'] == matched_beer_name:
                return beer
    else:
        return {"error": "No similar beer found. Please check the name."}

# Routes for Image Retrieval API
@app.get("/retrieve-images/{folder_name}")
def retrieve_images(folder_name: str):
    """
    Retrieve the URLs for 'poster.jpeg' and 'logo.jpeg' from a specific folder.
    """
    # Construct the path to the folder
    folder_path = BASE_DIR / folder_name

    # Check if the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")

    # Construct paths to the images
    poster_path = folder_path / "poster.jpeg"
    logo_path = folder_path / "logo.jpeg"

    # Check if both images exist in the folder
    missing_files = []
    if not poster_path.exists():
        missing_files.append("poster.jpeg")
    if not logo_path.exists():
        missing_files.append("logo.jpeg")

    if missing_files:
        raise HTTPException(
            status_code=404,
            detail=f"Missing files: {', '.join(missing_files)} in folder '{folder_name}'",
        )

    # Return URLs to access the images
    return {
        "poster_image_url": f"http://127.0.0.1:8000/image/{folder_name}/poster.jpeg",
        "logo_image_url": f"http://127.0.0.1:8000/image/{folder_name}/logo.jpeg",
    }

@app.get("/image/{folder_name}/{image_name}")
def get_image(folder_name: str, image_name: str):
    """
    Serve an image file given its folder name and file name.
    """
    # Construct the full path to the requested image
    image_path = BASE_DIR / folder_name / image_name

    # Validate the image path
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Image '{image_name}' not found in folder '{folder_name}'"
        )

    # Serve the image file as a response
    return FileResponse(str(image_path))

# Define the prediction API endpoint
@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = await read_image(file)
        image_tensor = transform(image).unsqueeze(0)

        # Get prediction and map to beer label
        predicted_class = predict(image_tensor)
        predicted_beer = label_mapping.get(predicted_class, "Unknown Beer")

        return {"prediction": predicted_beer}
    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the IBeer.ai API! Use '/beer/{beer_name}' to get beer details, '/retrieve-images/{folder_name}' to get image details, and '/predict' to classify beer images."}
