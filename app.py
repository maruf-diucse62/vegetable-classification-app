import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import model_download

# Download model
model_download.download_model()

# Load model
model = torch.load("best_model.pth", map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("Vegetable Classification App")
st.write("Upload an image to classify the vegetable")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted = output.argmax(1).item()

    st.success(f"Predicted Class: **{predicted}**")
