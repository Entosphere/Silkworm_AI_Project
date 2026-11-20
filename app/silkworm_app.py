
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Silkworm AI",
    page_icon="🐛",
    layout="centered"
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobilenetv3_silkworm_baseline.pth"

DEVICE = torch.device("cpu")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes, image_size


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(180),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return tensor


# ---------------- PREDICTION ----------------

def predict_image(image, model, classes, image_size):
    tensor = preprocess_image(image, image_size)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = classes[predicted_index.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score, probabilities[0].tolist()


# ---------------- APPLICATION INTERFACE ----------------

st.title("🐛 Silkworm AI")
st.subheader("AI-Based Silkworm Health Screening")

st.write(
    "Upload a silkworm image to receive an AI-generated "
    "classification of its apparent health status."
)

st.info(
    "Current model: Healthy vs. Unhealthy (non-fungal). "
    "Specific disease identification is not available yet."
)

uploaded_file = st.file_uploader(
    "Upload a silkworm image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Silkworm Image",
        use_container_width=True
    )

    if st.button("Analyse Image", type="primary"):

        try:
            with st.spinner("Analysing image..."):

                model, classes, image_size = load_model()

                predicted_class, confidence, probabilities = predict_image(
                    image,
                    model,
                    classes,
                    image_size
                )

            st.divider()
            st.header("Analysis Result")

            if predicted_class == "healthy":
                st.success("Prediction: HEALTHY")
            else:
                st.warning("Prediction: UNHEALTHY (NON-FUNGAL)")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.subheader("Class Probabilities")

            for class_name, probability in zip(classes, probabilities):
                st.write(f"**{class_name.replace('_', ' ').title()}**")
                st.progress(float(probability))
                st.caption(f"{probability * 100:.2f}%")

            st.caption(
                "Confidence is the model's prediction score, "
                "not a measure of diagnostic certainty."
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")

st.divider()

st.caption(
    "Silkworm AI | Experimental educational project | "
    "Results require expert verification."
)
