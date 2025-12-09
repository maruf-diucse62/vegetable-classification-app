import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# ----------------------------
# 1. Streamlit App Settings
# ----------------------------
st.set_page_config(
    page_title="Vegetable Classification App",
    page_icon="🥦",
    layout="centered"
)
st.title("🥦 Vegetable Classification App")
st.write("Upload an image of a vegetable and get its class prediction.")

# ----------------------------
# 2. Download model from Google Drive
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
MODEL_PATH = "best_vegetable_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

# ----------------------------
# 3. Define class names
# ----------------------------
class_names = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

# ----------------------------
# 4. Define your custom CNN architecture
# ----------------------------
# -----> PLACE YOUR KAGGLE MODEL CLASS HERE <-----
# Example placeholder:
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace with your original CNN layers from Kaggle
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(16 * 112 * 112, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(class_names))
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# ------------------------------------------------

# ----------------------------
# 5. Load model
# ----------------------------
model = CustomCNN()
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model = checkpoint  # in case full model was saved

model.eval()
st.success("Model loaded successfully!")

# ----------------------------
# 6. Image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# 7. Streamlit Image Upload & Prediction
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Image (JPG, PNG, JPEG)", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted_idx = output.argmax(1).item()
        predicted_class = class_names[predicted_idx]
        probabilities = torch.softmax(output, dim=1)[0]

    # Display prediction
    st.markdown(f"""
        <div style='padding:10px; border-radius:10px; background-color:#E6F4EA'>
            <h3 style='color:#2E7D32;'>Predicted Class: {predicted_class}</h3>
        </div>
    """, unsafe_allow_html=True)

    # Display top 3 predictions with confidence
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    st.write("Top 3 predictions:")
    for i, idx in enumerate(top3_idx):
        st.write(f"{i+1}. {class_names[idx]} — {top3_prob[i]*100:.2f}%")

# ----------------------------
# 8. Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed by Md. Abdullah Al Maruf | Powered by PyTorch & Streamlit")
