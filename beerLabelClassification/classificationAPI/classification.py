from fastapi import FastAPI, UploadFile, HTTPException
import shutil
import os
import torch
from model import load_model
from utils import predict_image
from config import MODEL_PATH

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, device)


@app.post("/predict")
async def predict(file: UploadFile):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        label = predict_image(model, temp_file_path, device)

        os.remove(temp_file_path)

        return {"Predicted Label": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
