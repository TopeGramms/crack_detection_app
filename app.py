import io
import base64
from datetime import datetime
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
  page_title="AI-Powered Concrete Crack Detection System",
  page_icon="🏗️",
  layout="wide",
)


# -----------------------------
# Custom CSS
# -----------------------------
def load_custom_css():
  css = """
  <style>
  :root{--accent-green:#16a34a; --accent-red:#f97316; --accent-dark:#0f172a; --card-bg:#ffffff; --muted:#6b7280}
  body { background-color: #f4f6f8; }
  .header { padding: 18px 24px; border-radius: 12px; background: linear-gradient(90deg, rgba(255,255,255,0.6), rgba(255,255,255,0.4)); box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
  .title { font-size:28px; font-weight:700; color:var(--accent-dark); }
  .subtitle { color:var(--muted); margin-top:6px; }
  .card { background: var(--card-bg); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(15,23,42,0.04); }
  .image-preview { border-radius:8px; overflow:hidden; border:1px solid rgba(15,23,42,0.04); }
  .big-alert { padding:20px; border-radius:12px; color: white; font-weight:700; text-align:center; }
  .progress-wrap { background: #e6e9ee; border-radius: 12px; padding:6px; }
  .progress-bar { height:18px; border-radius: 10px; }
  .muted { color:var(--muted); }
  .footer { color:var(--muted); font-size:13px; padding:12px; text-align:center; }
  .sample-img { border-radius:8px; }
  </style>
  """
  st.markdown(css, unsafe_allow_html=True)


# -----------------------------
# Utilities: model loading, preprocessing, predict, report
# -----------------------------
MODEL_PATH = "crack_detection_model.keras"


def _st_cache_resource(func):
  # compatibility wrapper: prefer st.cache_resource if available
  try:
    return st.cache_resource(func)
  except Exception:
    return st.cache(allow_output_mutation=True)(func)


@_st_cache_resource
def load_model(path: str):
  """Load and cache the Keras model."""
  try:
    model = tf.keras.models.load_model(path)
    return model
  except Exception as e:
    st.error(f"Failed to load model: {e}")
    return None


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
  img = image.convert("RGB").resize(target_size)
  arr = np.asarray(img).astype("float32") / 255.0
  arr = np.expand_dims(arr, axis=0)
  return arr


def predict_image(model, image: Image.Image):
  """Return (label, confidence, raw_score)"""
  try:
    x = preprocess_image(image)
    score = float(model.predict(x, verbose=0)[0][0])
    # Assuming model outputs probability of 'crack'
    if score >= 0.5:
      label = "Crack Detected"
      confidence = score
    else:
      label = "No Crack Detected"
      confidence = 1.0 - score
    return label, confidence, score
  except Exception as e:
    raise RuntimeError(f"Prediction failed: {e}")


def generate_report(pred_label, confidence, score, model_name, timestamp):
  txt = (
    f"AI-Powered Concrete Crack Detection Report\n"
    f"Generated: {timestamp}\n"
    f"Model: {model_name}\n"
    f"Result: {pred_label}\n"
    f"Confidence: {confidence * 100:.2f}%\n"
    f"Raw Score (model output): {score:.4f}\n"
  )
  return txt


# -----------------------------
# Sample image generation (placeholder images)
# -----------------------------
def create_sample_image(text: str, size=(600, 400), bgcolor=(240, 243, 246)) -> Image.Image:
  img = Image.new("RGB", size, bgcolor)
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.truetype("arial.ttf", 36)
  except Exception:
    font = ImageFont.load_default()
  w, h = draw.textsize(text, font=font)
  draw.text(((size[0] - w) / 2, (size[1] - h) / 2), text, fill=(32, 35, 41), font=font)
  # add subtle crack-like line for 'Crack' sample
  if "Crack" in text:
    draw.line((80, 100, 520, 300), fill=(80, 40, 30), width=6)
    draw.line((120, 60, 480, 260), fill=(120, 60, 40), width=3)
  return img


def pil_image_to_bytes(img: Image.Image, fmt="PNG"):
  buf = io.BytesIO()
  img.save(buf, format=fmt)
  byte_im = buf.getvalue()
  return byte_im


# -----------------------------
# App Layout and Logic
# -----------------------------
def main():
  load_custom_css()

  # Header
  with st.container():
    st.markdown(
      "<div class='header'><div class='title'>🏗️ AI-Powered Concrete Crack Detection System</div>"
      "<div class='subtitle'>Upload a concrete surface image and the MobileNetV2-based classifier will detect cracks and provide confidence.</div></div>",
      unsafe_allow_html=True,
    )

  # Sidebar
  with st.sidebar:
    st.header("Project Overview")
    st.write("Final-year project demo — MobileNetV2 transfer learning for crack detection.")
    st.markdown("---")
    st.subheader("Instructions")
    st.write("- Upload an image (jpg, jpeg, png).\n- Or click a sample image to try instantly.\n- Press Analyze to run the model.")
    st.markdown("---")
    st.subheader("Model Info")
    st.write("Architecture: MobileNetV2 (transfer learning)")
    model_status_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Dataset & Metrics")
    st.write("Dataset: Custom concrete crack dataset (placeholder)")
    st.metric("Accuracy (placeholder)", "-- %")
    st.markdown("---")
    st.subheader("Developer")
    st.write("Student: Your Name Here\nEmail: you@example.com")

  # Load model (cached)
  model = load_model(MODEL_PATH)
  model_loaded = model is not None
  if model_loaded:
    model_status_placeholder.success("Model loaded: MobileNetV2 (transfer learning)")
  else:
    model_status_placeholder.error("Model not loaded — check model path")

  # Main two-column layout
  left_col, right_col = st.columns([2, 1], gap="large")

  # Left column: upload + sample images
  with left_col:
    st.markdown("### Image Input")
    uploaded_file = st.file_uploader("Drag & drop or click to upload", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # Sample images
    st.markdown("### Try Sample Images")
    sample1 = create_sample_image("Crack Example")
    sample2 = create_sample_image("No Crack Example")
    s1_bytes = pil_image_to_bytes(sample1)
    s2_bytes = pil_image_to_bytes(sample2)

    cols = st.columns(2)
    with cols[0]:
      st.image(s1_bytes, caption="Crack Example", use_column_width=True, output_format="PNG")
      if st.button("Use Crack Sample"):
        st.session_state['selected_image'] = sample1
        st.session_state['auto_predict'] = True
    with cols[1]:
      st.image(s2_bytes, caption="No Crack Example", use_column_width=True, output_format="PNG")
      if st.button("Use No-Crack Sample"):
        st.session_state['selected_image'] = sample2
        st.session_state['auto_predict'] = True

    st.markdown("---")
    # Image preview card
    st.markdown("### Image Preview")
    preview_card = st.container()
    selected_image = None
    if 'selected_image' in st.session_state and st.session_state.get('selected_image') is not None:
      selected_image = st.session_state.get('selected_image')

    if uploaded_file is not None:
      try:
        img = Image.open(uploaded_file).convert('RGB')
        selected_image = img
        st.session_state['selected_image'] = img
      except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")

    if selected_image is not None:
      with preview_card:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(selected_image, use_column_width=True, caption="Selected Image")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
      st.info("No image selected yet — upload or pick a sample.")

  # Right column: prediction results and controls
  with right_col:
    st.markdown("### Prediction")
    result_card = st.container()
    analyze = st.button("Analyze Image")

    # Determine whether to auto run prediction
    auto_run = st.session_state.get('auto_predict', False)

    if auto_run:
      analyze = True
      st.session_state['auto_predict'] = False

    if analyze and (st.session_state.get('selected_image', None) is not None):
      image = st.session_state['selected_image']
      if not model_loaded:
        st.error("Model not available. Cannot run prediction.")
      else:
        with st.spinner("Analyzing image..."):
          try:
            label, confidence, raw_score = predict_image(model, image)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Result card
            if "Crack" in label:
              color = "#f97316"
              icon = "⚠️ Crack Detected"
            else:
              color = "#16a34a"
              icon = "✅ No Crack Detected"

            st.markdown(f"<div class='big-alert' style='background:{color}'> {icon} </div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.progress(min(int(confidence * 100), 100))

            # Downloadable report
            report_txt = generate_report(label, confidence, raw_score, "MobileNetV2 (transfer learning)", timestamp)
            st.download_button("Download Report", data=report_txt, file_name=f"crack_report_{timestamp.replace(' ', '_')}.txt")

            # store last prediction
            st.session_state['last_prediction'] = {
              'label': label,
              'confidence': confidence,
              'raw_score': raw_score,
              'timestamp': timestamp,
            }

          except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
      st.info("No prediction run yet. Upload or select an image, then click Analyze Image.")


if __name__ == "__main__":
  main()
