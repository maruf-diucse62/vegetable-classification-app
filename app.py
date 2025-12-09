import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import gdown

# ----------------------------
# 1. App Settings
# ----------------------------
st.set_page_config(
    page_title="Vegetable Classification App",
    page_icon="🥦",
    layout="wide"
)

# Sidebar
st.sidebar.title("🥦 Vegetable Classification")
st.sidebar.write(
    """
    This app classifies vegetables into 15 classes:
    Bean, Bitter_Gourd, Bottle_Gourd, Brinjal, Broccoli,
    Cabbage, Capsicum, Carrot, Cauliflower, Cucumber,
    Papaya, Potato, Pumpkin, Radish, Tomato.
    
    Upload one or multiple images to see predictions.
    """
)

# ----------------------------
# 2. Model Download
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1kodkQrwwJxuRGEKHgo2P7zU0o4P0pk1l"
MODEL_PATH = "full_vegetable_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

# ----------------------------
# 3. Class Names & Bengali Mapping
# ----------------------------
class_names = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

bengali_names = {
    "Bean": "শিম",
    "Bitter_Gourd": "করলা",
    "Bottle_Gourd": "লাউ",
    "Brinjal": "বেগুন",
    "Broccoli": "ব্রকলি",
    "Cabbage": "বাঁধাকপি",
    "Capsicum": "শিমলমরিচ",
    "Carrot": "গাজর",
    "Cauliflower": "ফুলকপি",
    "Cucumber": "শসা",
    "Papaya": "পেপে",
    "Potato": "আলু",
    "Pumpkin": "কুমড়া",
    "Radish": "মূলা",
    "Tomato": "টমেটো"
}

# ----------------------------
# 4. Load Full Model (weights_only=False)
# ----------------------------
try:
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------------------
# 5. Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# 6. File Uploader (Single or Multiple Images)
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload Image(s) (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_t)
            predicted_idx = output.argmax(1).item()
            predicted_class = class_names[predicted_idx]
            bengali_name = bengali_names.get(predicted_class, "")
            probabilities = torch.softmax(output, dim=1)[0]

        # Prediction box
        st.markdown(f"""
            <div style='padding:10px; border-radius:10px; background-color:#E6F4EA'>
                <h3 style='color:#2E7D32;'>Predicted Class: {predicted_class} ({bengali_name})</h3>
            </div>
        """, unsafe_allow_html=True)

        # Top-3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        st.write("Top 3 predictions:")
        for i, idx in enumerate(top3_idx):
            class_eng = class_names[idx]
            class_bn = bengali_names.get(class_eng, "")
            st.write(f"{i+1}. {class_eng} ({class_bn}) — {top3_prob[i]*100:.2f}%")

# ----------------------------
# 7. Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed by Md Abdullah Al Maruf | Powered by PyTorch & Streamlit")
