import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# வெப்சைட் செட்டிங்ஸ்
st.set_page_config(page_title="Deepfake Detection", page_icon="🛡️", layout="centered")
st.title("🛡️ AI Deepfake Detection System")
st.write("உங்களுடைய படம் உண்மையானதா அல்லது போலியானதா என கண்டறியவும்.")

# லோக்கல் ஃபோல்டரில் இருக்கும் மாடலை லோடு செய்கிறோம்
MODEL_PATH = "efficientnet_deepfake_best.keras"

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        # compile=False போடுவதால் quantization_config எரர் வராது
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    else:
        st.error("மாடல் ஃபைலை ஃபோல்டரில் காணவில்லை!")
        return None

model = load_my_model()

if model is not None:
    st.success("AI மாதிரி வெற்றிகரமாக ஏற்றப்பட்டது!")
    
    # ஃபைல் அப்லோடர்
    uploaded_file = st.file_uploader("ஒரு படத்தை தேர்ந்தெடுக்கவும்...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='பதிவேற்றப்பட்ட படம்', use_container_width=True)
        
        if st.button("Check Result"):
            with st.spinner("பகுப்பாய்வு செய்யப்படுகிறது..."):
                # Preprocessing (EfficientNet-க்கு ஏத்தபடி மாற்றுவது)
                img = np.array(image.convert('RGB'))
                img = cv2.resize(img, (380, 380))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Prediction
                prediction = model.predict(img)[0][0]
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                label = "REAL" if prediction > 0.5 else "FAKE"
                
                st.write("---")
                if label == "REAL":
                    st.success(f"📊 **முடிவு:** இந்த படம் **நிஜமானது (REAL)** - நிச்சயம்: {confidence*100:.2f}%")
                else:
                    st.error(f"🚨 **முடிவு:** எச்சரிக்கை! இது **போலி படம் (FAKE)** - நிச்சயம்: {confidence*100:.2f}%")