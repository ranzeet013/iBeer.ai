import torch
import torch.nn as nn
from torchvision import models
from config import MODEL_PATH, DEVICE

class BeerModel(nn.Module):
    def __init__(self):
        super(BeerModel, self).__init__()
        # ssing a pre-trained VGG16 model for transfer learning
        self.network = models.vgg16(pretrained=True)
        
        # freeze all layers except the final classifier
        for param in self.network.parameters():
            param.requires_grad = False
        
        self.network.classifier[6] = nn.Linear(self.network.classifier[6].in_features, 10)  # 10 classes

    def forward(self, x):
        return self.network(x)

model = None

# load model function
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

# image prediction function
def predict(image_tensor):
    model = load_model()
    with torch.no_grad():  
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  
        return predicted.item()
