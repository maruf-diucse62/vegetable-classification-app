import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# ----------------------------
# 1. Download model from Google Drive
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
MODEL_PATH = "best_vegetable_model.pth"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ----------------------------
# 2. Define your model architecture
# ----------------------------
class VegetableModel(nn.Module):
    def __init__(self):
        super(VegetableModel, self).__init__()
        # Replace this with your actual architecture
        self.fc = nn.Linear(224*224*3, 10)  # Example: 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# 3. Load the checkpoint
# ----------------------------
model = VegetableModel()
class_names = [str(i) for i in range(10)]  # Default class names

try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_names = checkpoint.get("class_names", class_names)
        st.info(f"Checkpoint loaded. Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        # If the .pth is a full model
        model = checkpoint
    
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ----------------------------
# 4. Image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.title("Vegetable Classification App")
st.write("Upload an image to classify the vegetable")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted_idx = output.argmax(1).item()
        predicted_class = class_names[predicted_idx]

    st.success(f"Predicted Class: **{predicted_class}**")
