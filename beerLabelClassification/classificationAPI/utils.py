import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from config import CLASSIFIED_LABELS_PATH

# Label mappings
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

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

    predicted_label = reverse_label_mapping[predicted_class]
    save_result(predicted_label, image_path)
    return predicted_label


def save_result(label, image_path):
    if not os.path.exists(CLASSIFIED_LABELS_PATH):
        os.makedirs(CLASSIFIED_LABELS_PATH)

    file_name = os.path.basename(image_path)
    output_file = os.path.join(CLASSIFIED_LABELS_PATH, f"{label}_{file_name}")

    with open(output_file, 'w') as f:
        f.write(f"Predicted Label: {label}")
