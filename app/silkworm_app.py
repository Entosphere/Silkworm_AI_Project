
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from uuid import uuid4

st.set_page_config(page_title="Silkworm AI", page_icon="🐛", layout="centered")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data_output"

DETECTOR_PATH = MODEL_DIR / "mobilenetv3_silkworm_detector.pth"
HEALTH_PATH = MODEL_DIR / "mobilenetv3_silkworm_baseline.pth"
DISEASE_PATH = MODEL_DIR / "mobilenetv3_model3_disease.pth"

CONFIDENCE_THRESHOLD = 70.0
DEVICE = torch.device("cpu")

FOLDERS = {
    "healthy": OUTPUT_DIR / "Healthy",
    "unhealthy_non_fungal": OUTPUT_DIR / "Unhealthy_Non_Fungal",
    "fungal": OUTPUT_DIR / "Fungal",
    "not_silkworm": OUTPUT_DIR / "Not_A_Silkworm",
    "grasserie": OUTPUT_DIR / "Grasserie",
    "pebrine": OUTPUT_DIR / "Pebrine",
    "flacherie": OUTPUT_DIR / "Flacherie",
    "unknown_uncertain": OUTPUT_DIR / "Unknown_Uncertain",
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)


def build_transform(size):
    return transforms.Compose([
        transforms.Resize(180),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_model(classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    return model


@st.cache_resource
def load_models():
    detector_ckpt = torch.load(
        DETECTOR_PATH, map_location=DEVICE, weights_only=True
    )
    health_ckpt = torch.load(
        HEALTH_PATH, map_location=DEVICE, weights_only=True
    )
    disease_ckpt = torch.load(
        DISEASE_PATH, map_location=DEVICE, weights_only=True
    )

    detector_classes = detector_ckpt["classes"]
    health_classes = health_ckpt["classes"]
    disease_classes = disease_ckpt["class_names"]

    detector = build_model(detector_classes)
    detector.load_state_dict(detector_ckpt["model_state_dict"])
    detector.eval()

    health = build_model(health_classes)
    health.load_state_dict(health_ckpt["model_state_dict"])
    health.eval()

    disease = build_model(disease_classes)
    disease.load_state_dict(disease_ckpt["model_state_dict"])
    disease.eval()

    return (
        detector, detector_classes, detector_ckpt["image_size"],
        health, health_classes, health_ckpt["image_size"],
        disease, disease_classes, disease_ckpt["image_size"]
    )


def predict(image, model, classes, image_size):
    tensor = build_transform(image_size)(
        image.convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = int(probabilities.argmax())

    return (
        classes[index],
        float(probabilities[index]) * 100,
        probabilities.tolist()
    )


def save_image(image_bytes, original_name, folder):
    extension = Path(original_name).suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        extension = ".jpg"

    path = folder / f"{uuid4().hex}{extension}"
    path.write_bytes(image_bytes)

    return path


def show_probabilities(classes, probabilities):
    for name, probability in zip(classes, probabilities):
        st.write(name.replace("_", " ").title())
        st.progress(float(probability))
        st.caption(f"{probability * 100:.2f}%")


st.title("🐛 Silkworm AI")
st.subheader("AI-Based Silkworm Detection and Disease Screening")

st.info(
    "Three-stage experimental pipeline: "
    "silkworm detection, health screening, and classification "
    "of three trained disease categories."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    st.divider()
    st.subheader("Manual fungal verification")

    manual_fungal = st.checkbox(
        "I have manually verified this image as fungal"
    )

    if st.button("Save manually verified fungal image"):
        if manual_fungal:
            save_image(
                image_bytes,
                uploaded.name,
                FOLDERS["fungal"]
            )
            st.success("Image saved in Fungal.")
        else:
            st.warning("Confirm manual verification first.")

    st.divider()

    if st.button("Analyse Image", type="primary"):
        try:
            with st.spinner("Loading models and analysing image..."):

                (
                    detector, detector_classes, detector_size,
                    health, health_classes, health_size,
                    disease, disease_classes, disease_size
                ) = load_models()

                st.header("Analysis Results")

                # STAGE 1: SILKWORM DETECTION
                st.write("### Stage 1 — Silkworm Detection")

                det_class, det_conf, _ = predict(
                    image,
                    detector,
                    detector_classes,
                    detector_size
                )

                st.metric("Detection confidence", f"{det_conf:.2f}%")

                if det_conf < CONFIDENCE_THRESHOLD:
                    st.warning(
                        "Uncertain detection. Manual review required."
                    )

                elif det_class == "not_silkworm":
                    save_image(
                        image_bytes,
                        uploaded.name,
                        FOLDERS["not_silkworm"]
                    )
                    st.error("Not identified as a silkworm.")
                    st.success("Saved in Not_A_Silkworm.")

                elif det_class == "silkworm":

                    st.success("Silkworm detected.")

                    # STAGE 2: HEALTH SCREENING
                    st.write("### Stage 2 — Health Screening")

                    health_class, health_conf, health_probs = predict(
                        image,
                        health,
                        health_classes,
                        health_size
                    )

                    st.metric(
                        "Health screening confidence",
                        f"{health_conf:.2f}%"
                    )

                    show_probabilities(health_classes, health_probs)

                    if health_conf < CONFIDENCE_THRESHOLD:
                        st.warning(
                            "Uncertain health result. Manual review required."
                        )

                    elif health_class == "healthy":
                        save_image(
                            image_bytes,
                            uploaded.name,
                            FOLDERS["healthy"]
                        )
                        st.success("Prediction: HEALTHY")
                        st.success("Saved in Healthy.")

                    elif health_class == "unhealthy_non_fungal":

                        st.warning("Unhealthy result detected.")

                        # STAGE 3: DISEASE CLASSIFICATION
                        st.write("### Stage 3 — Disease Classification")

                        disease_class, disease_conf, disease_probs = predict(
                            image,
                            disease,
                            disease_classes,
                            disease_size
                        )

                        st.metric(
                            "Disease classification confidence",
                            f"{disease_conf:.2f}%"
                        )

                        show_probabilities(
                            disease_classes,
                            disease_probs
                        )

                        if disease_conf < CONFIDENCE_THRESHOLD:
                            destination = "unknown_uncertain"
                            st.warning(
                                "Low-confidence disease result. "
                                "Classified as Unknown / Uncertain."
                            )

                        else:
                            normalized_class = disease_class.lower()

                            if normalized_class in [
                                "grasserie", "pebrine", "flacherie"
                            ]:
                                destination = normalized_class
                                st.success(
                                    f"Predicted category: {disease_class.title()}"
                                )
                            else:
                                destination = "unknown_uncertain"
                                st.warning(
                                    "Unrecognized disease class. "
                                    "Manual review required."
                                )

                        save_image(
                            image_bytes,
                            uploaded.name,
                            FOLDERS[destination]
                        )

                        st.success(
                            f"Image saved in {FOLDERS[destination].name}."
                        )

                    else:
                        save_image(
                            image_bytes,
                            uploaded.name,
                            FOLDERS["unknown_uncertain"]
                        )
                        st.warning(
                            "Unknown health class. Saved for manual review."
                        )

                else:
                    st.warning(
                        "Unknown detector class. Manual review required."
                    )

        except Exception as error:
            st.error(f"An error occurred: {error}")


st.divider()
st.subheader("Output Folder Counts")

for folder in FOLDERS.values():
    count = sum(1 for p in folder.iterdir() if p.is_file())
    st.write(f"**{folder.name}:** {count} images")


st.caption(
    "Silkworm AI is an experimental educational prototype, "
    "not a veterinary diagnostic system. Disease classification "
    "is limited to the three categories used during Model 3 training. "
    "Expert verification is required."
)
