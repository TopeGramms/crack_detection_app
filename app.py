import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Page config: centered layout
st.set_page_config(page_title="Concrete Crack Detection v2", layout="centered")


# Cached model and threshold loader
@st.cache_resource
def load_model(path: str):
    """Load and return a Keras model from the given path."""
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_threshold(path: str):
    """Load optimal threshold from numpy file."""
    return float(np.load(path))


def preprocess(img: Image.Image, size=(224, 224)) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image, threshold: float):
    """Run model prediction and return (is_crack: bool, confidence: float, raw_score: float).
    Uses optimal threshold instead of hardcoded 0.5.
    """
    x = preprocess(img)
    pred = model.predict(x, verbose=0)
    score = float(np.ravel(pred)[0])
    is_crack = score >= threshold
    confidence = score if is_crack else 1.0 - score
    return is_crack, confidence, score


def main():
    st.title("Concrete Crack Detection v2")

    # Load model and threshold (cached)
    try:
        model = load_model("crack_detection_model_v2.keras")
        threshold = load_threshold("optimal_threshold.npy")
    except Exception as e:
        st.error(f"Failed to load model/threshold: {e}")
        return

    # File uploader
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        # Show uploaded image
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        # Run prediction
        try:
            is_crack, confidence, raw_score = predict(model, img, threshold)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        pct = confidence * 100

        # Display result clearly and show numeric confidence
        if is_crack:
            st.error("⚠️ Crack Detected")
        else:
            st.success("✅ No Crack Detected")

        # Numeric confidence metric and progress bar
        st.metric(label="Confidence", value=f"{pct:.2f}%")
        st.progress(min(int(pct), 100))

        # Debug info
        with st.expander("📊 Debug Info"):
            st.write(f"Raw Score: {raw_score:.4f}")
            st.write(f"Threshold: {threshold:.4f}")

    # Helpful short note
    st.write("---")
    st.write("Note: Model trained on 40,000 images with data augmentation. For best results use clear photos with good lighting.")


if __name__ == "__main__":
    main()