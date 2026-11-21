# -*- coding: utf-8 -*-
import os
import zipfile
import glob
import gdown
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

st.set_page_config(page_title="Melanoma Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Melanoma Cancer Detection")
st.write("Upload a skin-lesion image and the CNN will classify it as **benign** or **malignant**.")

st.warning(
    "This is an educational AI model and is not a medical diagnosis. "
    "Please consult a qualified healthcare professional for any concerning skin lesion."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

zip_path = os.path.join(DATA_DIR, "archive_3.zip")
extract_path = os.path.join(DATA_DIR, "extracted_data")
model_save_path = os.path.join(DATA_DIR, "melanoma_cnn_model.pth")

GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1BCASrPCTr33zU2HaK1od5MMZyG5i_oxd/view?usp=sharing"

# Keep the original CNN architecture unchanged.
class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class MelanomaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_mapping = {"benign": 0, "malignant": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, "path"]).convert("RGB")
        label = self.label_mapping[self.df.loc[idx, "label"]]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

@st.cache_resource(show_spinner="Preparing the CNN model...")
def load_or_train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelanomaCNN().to(device)

    # Reuse an already trained model if one exists.
    if os.path.exists(model_save_path):
        try:
            state = torch.load(model_save_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model, device, "loaded"
        except Exception:
            pass

    # Download/extract the dataset only when a trained model is not available.
    if not os.path.exists(zip_path):
        gdown.download(url=GOOGLE_DRIVE_URL, output=zip_path, quiet=False)

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    dataset_dir = os.path.join(extract_path, "melanoma_cancer_dataset")
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    image_data = []
    for img_path in all_images:
        parts = img_path.split(os.sep)
        if len(parts) >= 3:
            label = parts[-2].lower()
            split_type = parts[-3].lower()
            if label in ("benign", "malignant"):
                image_data.append({"path": img_path, "label": label, "split": split_type})

    df_images = pd.DataFrame(image_data)
    if df_images.empty:
        raise RuntimeError("No benign/malignant images were found in the dataset.")

    # Prefer the original train/test folders. Fall back to all data if needed.
    train_df = df_images[df_images["split"] == "train"]
    test_df = df_images[df_images["split"] == "test"]

    if train_df.empty:
        train_df = df_images.sample(frac=0.8, random_state=42)
    if test_df.empty:
        test_df = df_images.drop(train_df.index)

    train_dataset = MelanomaDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Same CNN training approach as the original project.
    for _ in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    model.eval()
    return model, device, "trained"

try:
    model, device, model_status = load_or_train_model()
except Exception as e:
    st.error(f"Could not prepare the CNN model: {e}")
    st.stop()

if model_status == "loaded":
    st.success("CNN model loaded successfully.")
else:
    st.success("CNN model trained and ready.")

st.divider()
st.subheader("📷 Upload an image")

uploaded_file = st.file_uploader(
    "Choose a skin-lesion image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔍 Detect Cancer", type="primary"):
        with st.spinner("Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

            labels = ["Benign", "Malignant"]
            prediction = labels[predicted_class]
            confidence = probabilities[predicted_class].item() * 100

        if prediction == "Malignant":
            st.error(f"Prediction: **{prediction}**")
        else:
            st.success(f"Prediction: **{prediction}**")

        st.metric("Model confidence", f"{confidence:.2f}%")
        st.caption(
            "The prediction is generated by the trained CNN and should not be used "
            "as a substitute for professional medical evaluation."
        )
else:
    st.info("Upload a JPG, JPEG, PNG, or BMP image to start detection.")
