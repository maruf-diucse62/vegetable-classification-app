import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Load model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=15)
    model.load_state_dict(torch.load("best_vegetable_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Vegetable class names
class_names = [
    "Bean", "Bitter Gourd", "Bottle Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

# UI
st.title("🥕 Vegetable Image Classification")
st.write("Upload an image and the model will predict the vegetable category.")

uploaded_file = st.file_uploader("Upload a vegetable image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = class_names[predicted.item()]

    st.success(f"### 🥦 Predicted Vegetable: **{result}**")
