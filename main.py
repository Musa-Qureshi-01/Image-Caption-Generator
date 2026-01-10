import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import time
import os
import base64


# ============================
# CONFIGURATION
# ============================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.keras")

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🧠",
    layout="wide"
)


# ============================
# CSS STYLING
# ============================
st.markdown("""
<style>

body {
    background-color: #0f1117;
}

h1 {
    font-size: 40px !important;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.caption-box {
    padding: 15px;
    border-radius: 12px;
    background: #1a1c23;
    border: 1px solid #333;
    color: #ffffff;
    font-size: 20px;
}

.step-box {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-left: 5px solid #00aaff;
    border-radius: 8px;
    margin-bottom: 12px;
}

.fade-in {
    animation: fadeIn 1s ease-in-out forwards;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(8px);}
    to {opacity: 1; transform: translateY(0px);}
}

.uploaded-image {
    transition: transform 0.3s ease;
}

.uploaded-image:hover {
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)



# ============================
# BASE64 IMAGE ENCODER
# ============================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


# ============================
# CAPTION GENERATION FUNCTION
# ============================
def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34):

    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    st.markdown("<div class='step-box fade-in'>📤 Extracting image features…</div>", unsafe_allow_html=True)
    time.sleep(0.5)
    image_features = feature_extractor.predict(arr, verbose=0)

    in_text = "startseq"
    tokens_list = []

    st.markdown("<div class='step-box fade-in'>🧠 Generating caption tokens…</div>", unsafe_allow_html=True)
    time.sleep(0.4)

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = caption_model.predict([image_features, seq], verbose=0)
        word_index = np.argmax(yhat)
        word = tokenizer.index_word.get(word_index)

        if word is None:
            break

        tokens_list.append(word)
        in_text += " " + word

        if word == "endseq":
            break

    caption = (
        in_text.replace("startseq", "")
               .replace("endseq", "")
               .strip()
    )

    return caption, tokens_list



# ============================
# MAIN STREAMLIT APP
# ============================
def main():

    st.title("🧠 AI Image Caption Generator")
    st.write("Upload an image and generate a smart description using Deep Learning.")

    # Sidebar model status
    with st.sidebar:
        st.header("⚙️ Model Status")

        st.success("✔ model.keras loaded") if os.path.exists(MODEL_PATH) else st.error("❌ model.keras missing")
        st.success("✔ tokenizer.pkl loaded") if os.path.exists(TOKENIZER_PATH) else st.error("❌ tokenizer.pkl missing")
        st.success("✔ feature_extractor.keras loaded") if os.path.exists(FEATURE_EXTRACTOR_PATH) else st.error("❌ feature_extractor.keras missing")

        st.markdown("---")
        st.info("Upload JPG / PNG to generate caption.")

    uploaded = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:

        with open("uploaded.jpg", "wb") as f:
            f.write(uploaded.getbuffer())

        st.markdown("<h4>Uploaded Image</h4>", unsafe_allow_html=True)

        img_base64 = get_base64_image("uploaded.jpg")

        # FIXED SIZE IMAGE DISPLAY (max-width)
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="{img_base64}" class="uploaded-image"
                     style="max-width:450px; width:100%; border-radius:12px; margin:auto;" />
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🔄 Processing…")
        time.sleep(0.4)

        caption, tokens = generate_caption(
            "uploaded.jpg", MODEL_PATH, TOKENIZER_PATH, FEATURE_EXTRACTOR_PATH
        )

        st.markdown("### 📝 Generated Caption")
        st.markdown(
            f"<div class='caption-box fade-in'>{caption}</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 🔍 Internal Working Breakdown")

        with st.expander("📌 Step-by-step Token Predictions"):
            st.write(tokens)

        with st.expander("🧩 How the Model Works"):
            st.write("""
            **1. CNN Feature Extraction (DenseNet201)**  
            Converts the image into 1920 essential features.

            **2. LSTM Decoder**  
            Predicts one word at a time based on image features + previous text.

            **3. Sequential Loop**  
            Continues predicting tokens until `endseq`.

            **4. Token Mapping**  
            Converts numeric outputs back into English words using the tokenizer.
            """)



if __name__ == "__main__":
    main()
