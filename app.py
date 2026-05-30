import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    TF_ERROR = None
except Exception as e:
    tf = None
    TF_ERROR = str(e)


st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 TruthLens")
st.write("Upload an image and the model will predict whether it is AI-generated or non-AI.")


MODULE_INFO = {
    "body": """Module 1: Body

The human body is one of the hardest things for AI to get right. Start with the hands — AI loves sneaking in an extra finger or two. From there, look at the teeth, the ears, and the hair. The skin may look too smooth, with no pores or texture. Eyes can look glassy or fake, and makeup may look painted on.""",

    "text": """Module 2: Text

AI cannot spell. Words on signs, t-shirts, menus, or book covers may look fine from far away, but become gibberish up close. Letters may blend together, flip backward, or become symbols that only look like letters.""",

    "reflections": """Module 3: Reflections

Shadows and reflections follow rules, but AI does not always know them. Shadows may point in different directions, mirrors may show impossible reflections, and water reflections may not match the real scene.""",

    "backgrounds": """Module 4: Backgrounds

The further from the center of an AI image, the weirder it gets. Objects near the edges may blend together, people may melt into backgrounds, and buildings or doors may look geometrically impossible.""",

    "textures": """Module 5: Textures

AI textures look good in a thumbnail but fall apart when zoomed in. Fabric patterns may not follow the body, seams may disappear, zippers may stop halfway, and everything may look too new with no scratches, stains, or wear.""",

    "metadata": """Module 6: Metadata / Technical

Real photos often carry hidden information such as camera model, lens, time, or location. AI images often have missing or suspicious metadata. Some AI tools may also leave behind software markers or recognizable style signatures.""",

    "objects": """Module 7: Objects

Small details are where AI gives up fastest. Look at the glasses — the arms that hook behind the ears often fade out or disappear. Jewelry may sink into the skin, necklaces may float, and watch faces may have melted numbers, wrong times, or blurry details."""
}


@st.cache_resource
def load_model():
    model_path = Path("keras_model.h5")
    labels_path = Path("labels.txt")

    if tf is None:
        return None, None, f"TensorFlow failed to load: {TF_ERROR}"

    if not model_path.exists():
        return None, None, "keras_model.h5 not found."

    if not labels_path.exists():
        return None, None, "labels.txt not found."

    model = tf.keras.models.load_model(model_path, compile=False)

    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if " " in line and line.split(" ", 1)[0].isdigit():
                line = line.split(" ", 1)[1]

            labels.append(line)

    return model, labels, None


def predict_image(model, labels, image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data, verbose=0)[0]

    index = np.argmax(prediction)
    label = labels[index]
    confidence = float(prediction[index])

    return label, confidence, prediction


def is_ai_label(label):
    label_lower = label.lower()

    if "non-ai" in label_lower or "nonai" in label_lower or "real" in label_lower:
        return False

    if "ai" in label_lower or "generated" in label_lower or "fake" in label_lower:
        return True

    return False


def get_module_info(label):
    label_lower = label.lower()

    for module_key, module_text in MODULE_INFO.items():
        if module_key in label_lower:
            return module_text

    return None


model, labels, error = load_model()

if error:
    st.error(error)
    st.stop()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    label, confidence, prediction = predict_image(model, labels, image)

    st.subheader("Detection Result")

    if is_ai_label(label):
        st.error("AI Generated")
    else:
        st.success("Non-AI / Real")

    st.write(f"**Predicted Label:** {label}")

    st.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )

    module_info = get_module_info(label)

    if module_info:
        st.subheader("Educational Explanation")
        st.info(module_info)
    else:
        st.subheader("Educational Explanation")
        st.warning("No matching educational module was found for this label.")

    with st.expander("Prediction Details"):
        for i, score in enumerate(prediction):
            st.write(f"{labels[i]}: {score * 100:.2f}%")