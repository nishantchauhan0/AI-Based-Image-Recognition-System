
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Recognition",
    page_icon="🖼️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cifar10_cnn.keras"
CLASS_PATH = BASE_DIR / "models" / "cifar10_class_names.txt"
RESULTS_PATH = BASE_DIR / "models" / "cifar10_comparison_results.json"

st.title("🖼️ AI-Based Image Recognition System")
st.caption(
    "CIFAR-10 Image Classification using CNN, "
    "with image predictions and model comparison."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "CIFAR-10 model files are missing. "
        "Please check the models folder."
    )
    st.stop()

try:
    model = load_model()
    class_names = CLASS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

tab1, tab2 = st.tabs(
    ["🔍 Image Recognition", "📊 Model Comparison"]
)

with tab1:
    st.subheader("Upload an Image")

    st.info(
        "CIFAR-10 classes: airplane, automobile, bird, cat, "
        "deer, dog, frog, horse, ship and truck."
    )

    uploaded_file = st.file_uploader(
        "Choose an image to classify",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns(2)

            with left:
                st.image(
                    image,
                    caption="Uploaded Image",
                    width="stretch"
                )

            with right:
                st.write("### Prediction")

                if st.button(
                    "Recognize Image",
                    type="primary"
                ):
                    with st.spinner("Analyzing image..."):
                        resized = image.resize((32, 32))
                        array = np.asarray(
                            resized,
                            dtype=np.float32
                        )
                        array = np.expand_dims(array, axis=0)

                        # The trained model applies rescaling internally.
                        predictions = model.predict(
                            array,
                            verbose=0
                        )[0]

                        best_index = int(
                            np.argmax(predictions)
                        )
                        label = class_names[best_index]
                        confidence = float(
                            predictions[best_index]
                        )

                    st.success(f"Predicted Class: {label.title()}")
                    st.metric(
                        "Confidence Score",
                        f"{confidence * 100:.2f}%"
                    )

                    st.write("### All Class Scores")

                    for name, score in zip(
                        class_names, predictions
                    ):
                        st.write(
                            f"**{name.title()}:** "
                            f"{score * 100:.2f}%"
                        )
                        st.progress(
                            float(np.clip(score, 0, 1))
                        )

                    st.caption(
                        "Confidence is the model's predicted score, "
                        "not a guarantee that the prediction is correct."
                    )

        except Exception as error:
            st.error(f"Could not process image: {error}")

with tab2:
    st.subheader("Model Accuracy Comparison")

    if RESULTS_PATH.exists():
        try:
            with open(
                RESULTS_PATH,
                encoding="utf-8"
            ) as file:
                results = json.load(file)

            names = list(results.keys())
            scores = [float(results[name]) for name in names]

            columns = st.columns(len(names))

            for column, name, score in zip(
                columns, names, scores
            ):
                column.metric(name, f"{score:.2f}%")

            fig, ax = plt.subplots()
            bars = ax.bar(names, scores)
            ax.set_title("CIFAR-10 Model Accuracy")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.bar_label(bars, fmt="%.2f%%")
            plt.xticks(rotation=15)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.error(f"Could not load comparison results: {error}")

    else:
        st.warning(
            "CIFAR-10 comparison results are not available yet. "
            "Your trained CNN test accuracy was 76.03%. "
            "We will add a fair comparison with another model next."
        )

st.divider()
st.caption(
    "Built with Python, TensorFlow, CNN, CIFAR-10, "
    "Streamlit and Matplotlib."
)
