import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import tempfile
from datetime import datetime
import requests
import json
from offline_translations import OFFLINE_TRANSLATIONS,OFFLINE_DATA

# Try to import Google GenAI - handle if not available
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ------------------------------
# Offline translations and precautions database
# ------------------------------



# ------------------------------
# Internet connectivity check
# ------------------------------
def check_internet_connection():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        return False

# ------------------------------
# Initialize Google GenAI client (only if available and online)
# ------------------------------
def initialize_genai_client():
    if GENAI_AVAILABLE and check_internet_connection():
        try:
            client = genai.Client(api_key="AIzaSyD5vWRUZG-ksss782D_AP85YsNUHUprrPg")  # Replace with your key
            return client
        except Exception:
            return None
    return None

# ------------------------------
# TensorFlow Model Prediction
# ------------------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model('trained_model.h5', compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ------------------------------
# Offline translation function
# ------------------------------
def get_offline_translation(disease, language):
    if language in OFFLINE_TRANSLATIONS and disease in OFFLINE_TRANSLATIONS[language]:
        return OFFLINE_TRANSLATIONS[language][disease]
    return disease  # Return original if translation not found

# ------------------------------
# Get precautions (offline)
# ------------------------------
def get_offline_precautions(disease, language):
    if disease in OFFLINE_DATA and language in OFFLINE_DATA[disease]:
        return OFFLINE_DATA[disease][language]
    return f"General care: Maintain proper plant hygiene, ensure adequate spacing, and monitor regularly for signs of disease."

# ------------------------------
# Initialize session state
# ------------------------------
for key in ["uploaded_file", "predicted_disease", "translated_prediction", 
            "chat_messages", "processing_message", "preferred_language", "language_changed", 
            "is_online", "genai_client", "offline_precautions_shown"]:
    if key not in st.session_state:
        if key == "chat_messages":
            st.session_state[key] = []
        elif key in ["processing_message", "language_changed", "offline_precautions_shown"]:
            st.session_state[key] = False
        elif key == "is_online":
            st.session_state[key] = check_internet_connection()
        elif key == "genai_client":
            st.session_state[key] = initialize_genai_client()
        else:
            st.session_state[key] = None

# ------------------------------
# Display connection status
# ------------------------------
if st.session_state.is_online:
    st.success("🌐 Online Mode: Full chat functionality available")
else:
    st.warning("📴 Offline Mode: Basic precautions,medicines and translations available")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Add refresh connection button in sidebar
if st.sidebar.button("🔄 Check Connection"):
    st.session_state.is_online = check_internet_connection()
    st.session_state.genai_client = initialize_genai_client()
    st.rerun()

# ------------------------------
# Home Page
# ------------------------------
if app_mode == "Home":
    st.header("🌿 Plant Disease Recognition System")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System!  
    Upload an image of a plant, and our system will detect any signs of disease.
    
    **Features:**
    - 🌐 **Online Mode**: Full chat functionality with AI assistant
    - 📴 **Offline Mode**: Basic disease detection with precautions,medicines and translations
    """)

# ------------------------------
# About Page
# ------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset has 87K RGB images of healthy and diseased crop leaves (38 classes).  
    - **Train:** 70K images  
    - **Validation:** 17K images  
    - **Test:** 33 images  
    
    #### Features
    - **Online Mode**: Real-time chat with AI assistant, dynamic translations
    - **Offline Mode**: Pre-loaded translations and precautions for major diseases
    """)

# ------------------------------
# Disease Recognition Page
# ------------------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")

    # Upload image
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    # Show image & predict
    if st.session_state.uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📷 Show Image"):
                st.image(st.session_state.uploaded_file, use_column_width=True)

        with col2:
            if st.button("🔍 Predict Disease"):
                with st.spinner("Please wait, predicting disease..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file.read())
                        tmp_path = tmp_file.name

                    result_index = model_prediction(tmp_path)
                    class_name = [
                        'Apple : Apple scab', 'Apple : Black rot', 'Apple : Cedar apple rust', 'Apple : healthy',
                        'Blueberry : healthy', 'Cherry : Powdery mildew', 'Cherry : healthy',
                        'Corn : Cercospora leaf spot Gray leaf spot', 'Corn : Common rust', 'Corn : Northern Leaf Blight',
                        'Corn : healthy', 'Grape : Black rot', 'Grape : Esca (Black Measles)',
                        'Grape : Leaf blight (Isariopsis Leaf Spot)', 'Grape : healthy',
                        'Orange : Huanglongbing (Citrus greening)',
                        'Peach : Bacterial spot', 'Peach : healthy', 'Pepper bell : Bacterial spot', 'Pepper bell : healthy',
                        'Potato : Early blight', 'Potato : Late blight', 'Potato : healthy', 'Raspberry : healthy',
                        'Soybean : healthy', 'Squash : Powdery mildew', 'Strawberry : Leaf scorch', 'Strawberry : healthy',
                        'Tomato : Bacterial spot', 'Tomato : Early blight', 'Tomato : Late blight', 'Tomato : Leaf Mold',
                        'Tomato : Septoria leaf spot', 'Tomato : Spider mites Two-spotted spider mite',
                        'Tomato : Target Spot', 'Tomato : Tomato Yellow Leaf Curl Virus',
                        'Tomato : Tomato mosaic virus', 'Tomato : healthy'
                    ]
                    st.session_state.predicted_disease = class_name[result_index]
                    st.session_state.translated_prediction = None
                    st.session_state.chat_messages = []
                    st.session_state.offline_precautions_shown = False

    # ------------------------------
    # Language selection
    # ------------------------------
    if st.session_state.predicted_disease:
        if st.session_state.preferred_language is None:
            st.markdown("### 🌐 Select Your Preferred Language")
            cols = st.columns(8)
            languages = ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Punjabi"]
            flags = ["🇺🇸","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳","🇮🇳"]
            for i, col in enumerate(cols):
                if col.button(f"{flags[i]} {languages[i]}"):
                    st.session_state.preferred_language = languages[i]
                    st.session_state.language_changed = True
                    st.rerun()
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🌐 Selected Language: **{st.session_state.preferred_language}**")
            with col2:
                if st.button("Change Language"):
                    st.session_state.preferred_language = None
                    st.session_state.language_changed = True
                    st.session_state.offline_precautions_shown = False
                    st.rerun()

    # ------------------------------
    # Handle translation and precautions (Online/Offline)
    # ------------------------------
    if st.session_state.language_changed and st.session_state.preferred_language:
        if st.session_state.is_online and st.session_state.genai_client:
            # Online mode - use GenAI for translation
            with st.spinner("Translating prediction and chat..."):
                try:
                    # Translate prediction
                    prompt_pred = f"Translate this plant disease prediction into {st.session_state.preferred_language}: '{st.session_state.predicted_disease}' give only the accurate translation, no extra words."
                    response_pred = st.session_state.genai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[types.Content(role="user", parts=[types.Part(text=prompt_pred)])]
                    )
                    st.session_state.translated_prediction = response_pred.text.strip()

                    # Translate chat messages
                    translated_chat = []
                    for msg in st.session_state.chat_messages:
                        prompt_msg = f"Translate this message into {st.session_state.preferred_language}: '{msg['text']}' give only accurate translation, no extra text."
                        response_msg = st.session_state.genai_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[types.Content(role="user", parts=[types.Part(text=prompt_msg)])]
                        )
                        translated_chat.append({"role": msg['role'], "text": response_msg.text.strip()})
                    st.session_state.chat_messages = translated_chat

                    st.session_state.language_changed = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
                    # Fallback to offline mode
                    st.session_state.translated_prediction = get_offline_translation(
                        st.session_state.predicted_disease, st.session_state.preferred_language
                    )
                    st.session_state.language_changed = False
        else:
            # Offline mode - use pre-loaded translations
            st.session_state.translated_prediction = get_offline_translation(
                st.session_state.predicted_disease, st.session_state.preferred_language
            )
            st.session_state.language_changed = False
            st.rerun()

    # ------------------------------
    # Show prediction
    # ------------------------------
    if st.session_state.predicted_disease and st.session_state.preferred_language:
        if st.session_state.translated_prediction:
            st.success(f"✅ Model Prediction: **{st.session_state.translated_prediction}**")
        else:
            st.success(f"✅ Model Prediction: **{st.session_state.predicted_disease}**")

    # ------------------------------
    # Show offline precautions if offline
    # ------------------------------
    if (not st.session_state.is_online and st.session_state.predicted_disease and 
        st.session_state.preferred_language and not st.session_state.offline_precautions_shown):
        
        st.markdown("---")
        st.subheader("🛡️ Disease Precautions & Treatment")
        
        precautions = get_offline_precautions(st.session_state.predicted_disease, st.session_state.preferred_language)
        
        st.info("📴 **Offline Mode**: Showing pre-loaded precautions and medicines")
        st.write("**Recommended Precautions:**")
        st.write(precautions["precaution"])
        st.write("**Recommended Medicines:**")
        st.write(precautions["medicine"])
        
        st.session_state.offline_precautions_shown = True

    # ------------------------------
    # Chat interface (Online only)
    # ------------------------------
    if (st.session_state.predicted_disease and st.session_state.preferred_language and 
        st.session_state.is_online and st.session_state.genai_client):
        
        st.markdown("---")
        st.subheader("💬 Chat with Agricultural Assistant")

        # Build chat HTML with timestamps & icons
        chat_html = '<div id="chat-container" style="height:350px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:15px; background-color:#fafafa; font-family:Arial, sans-serif; font-size:14px;">'
        for msg in st.session_state.chat_messages:
            time_str = datetime.now().strftime("%H:%M")
            if msg["role"] == "user":
                chat_html += f"""
                <div style="display:flex; justify-content:flex-end; margin:5px 0;">
                    <div style="background-color:#dcf8c6; padding:10px; border-radius:15px; max-width:70%; word-wrap:break-word;">
                        👨‍🌾 {msg['text']}<br><span style="font-size:10px; color:#555;">{time_str}</span>
                    </div>
                </div>
                """
            else:
                chat_html += f"""
                <div style="display:flex; justify-content:flex-start; margin:5px 0;">
                    <div style="background-color:#f1f0f0; padding:10px; border-radius:15px; max-width:70%; word-wrap:break-word;">
                        🤖 {msg['text']}<br><span style="font-size:10px; color:#555;">{time_str}</span>
                    </div>
                </div>
                """
        chat_html += '<div id="chat-end"></div></div>'
        chat_html += """
        <script>
        const chatContainer = document.getElementById("chat-container");
        const chatEnd = document.getElementById("chat-end");
        if(chatContainer && chatEnd){ chatEnd.scrollIntoView({behavior:"smooth"}); }
        </script>
        """

        components.html(chat_html, height=360, scrolling=False)

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask about precautions or treatment:", placeholder="e.g., How can I treat this disease?")
            submit_button = st.form_submit_button("Send")
            if submit_button and user_input.strip():
                st.session_state.chat_messages.append({"role": "user", "text": user_input.strip()})
                st.session_state.processing_message = True
                st.rerun()

        # Process API (Online only)
        if st.session_state.processing_message:
            with st.spinner("Assistant is typing..."):
                try:
                    response = st.session_state.genai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[types.Content(
                            role="model",
                            parts=[types.Part(text=f"You are an agricultural assistant. Give short, precise answers in {st.session_state.preferred_language}. related to {st.session_state.predicted_disease}")]
                        )] + [
                            types.Content(
                                role="model" if msg["role"]=="model" else "user",
                                parts=[types.Part(text=msg["text"])]
                            )
                            for msg in st.session_state.chat_messages
                        ]
                    )
                    reply = response.text.strip()
                    st.session_state.chat_messages.append({"role":"model", "text":reply})
                    st.session_state.processing_message = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_messages.append({
                        "role": "model",
                        "text": "Sorry, I encountered an error. Please try again."
                    })
                    st.session_state.processing_message = False
                    st.rerun()

    elif st.session_state.predicted_disease and st.session_state.preferred_language and not st.session_state.is_online:
        st.markdown("---")
        st.info("💬 **Chat feature requires internet connection.** Currently showing offline precautions and medicines above.")

