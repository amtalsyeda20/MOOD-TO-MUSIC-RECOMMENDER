import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser
from PIL import Image
import time

# 1. Page Configuration
st.set_page_config(page_title="Sentibeat AI", page_icon="🎧", layout="wide")

# Initialize Session State for the music button to work across clicks
if 'detected_mood' not in st.session_state:
    st.session_state.detected_mood = None
if 'detected_url' not in st.session_state:
    st.session_state.detected_url = None

# --- REUSABLE PREDICTION & ROCKING SCANNER ---
def process_vibe(image_data, model, labels, playlists):
    with st.status("🚀 INITIATING DEEP SCAN...", expanded=True) as status:
        st.write("📡 Accessing neural weights...")
        time.sleep(0.5)
        st.write("🧠 Analyzing pixel-level micro-expressions...")
        
        img = Image.open(image_data)
        cv2_img = np.array(img.convert('RGB'))
        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray_img, (48, 48))
        normalized_img = resized_img / 255.0
        reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))

        prediction = model.predict(reshaped_img)
        mood = labels[np.argmax(prediction)]
        conf = np.max(prediction) * 100
        
        time.sleep(0.5)
        st.write("✅ Vibe identified. Mapping to Bollywood frequency...")
        status.update(label="SCAN COMPLETE", state="complete", expanded=False)

    # Store results in session state so the button doesn't "break"
    st.session_state.detected_mood = mood
    st.session_state.detected_url = playlists.get(mood)

# 2. Premium CSS with "Popping" Animations
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* Animation for Popping Boxes */
    @keyframes popIn {
        0% { opacity: 0; transform: translateY(30px) scale(0.9); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    .info-card {
        background-color: #111;
        padding: 25px;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 20px;
        border-radius: 5px;
        animation: popIn 0.8s ease-out forwards;
        opacity: 0; /* Starts hidden for the animation */
    }

    /* Staggered delays for the pop-in effect */
    .card-1 { animation-delay: 0.2s; }
    .card-2 { animation-delay: 0.4s; }
    .card-3 { animation-delay: 0.6s; }
    .card-4 { animation-delay: 0.8s; }

    .brand-header {
        font-family: 'Courier New', Courier, monospace;
        font-size: 50px !important;
        font-weight: 900;
        letter-spacing: 10px;
        color: #ff4b4b;
        margin-top: -50px;
        margin-bottom: 30px;
    }

    .hero-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 80px !important;
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 0.9;
    }
    
    /* Rocking Button Effect */
    .stButton>button {
        border: 2px solid #ff4b4b !important;
        background-color: transparent !important;
        color: white !important;
        font-weight: bold !important;
        height: 60px;
        font-size: 18px !important;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background-color: #ff4b4b !important;
        box-shadow: 0px 0px 25px #ff4b4b;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_my_model():
    return load_model('my_mood_model.h5')

try:
    my_model = load_my_model()
    mood_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
except:
    st.error("SYSTEM ERROR: Model File Missing.")

# 4. Sidebar
with st.sidebar:
    st.markdown('<p style="color:#ff4b4b; font-weight:bold; letter-spacing:3px;">SENTIBEAT AI</p>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    app_mode = st.radio("NAVIGATE", ["EXPLORE HOME", "LIVE SCANNER", "PHOTO ANALYSIS"])

# 5. Playlists
mood_playlists = {
    "Happy": "https://www.youtube.com/watch?v=nJZcbidTutE", 
    "Sad": "https://www.youtube.com/watch?v=NvnBvjL87B0",
    "Neutral": "https://www.youtube.com/watch?v=Cb6wuzOurPc", 
    "Angry": "https://www.youtube.com/watch?v=abiL84EAWSY",
    "Surprise": "https://www.youtube.com/watch?v=M03GOY5eINg", 
    "Fear": "https://www.youtube.com/watch?v=D6MOuX980gc",
    "Disgust": "https://www.youtube.com/watch?v=Vng5mg0iY0Q"
}

# --- PAGE: HOME ---
if app_mode == "EXPLORE HOME":
    st.markdown('<p class="brand-header">SENTIBEAT AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">TRANSFORMING<br>EMOTION INTO BEATS.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card card-1">
            <h4 style="color:#ff4b4b;">01 // NEURAL FACIAL RECOGNITION</h4>
            <p>Our deep learning architecture processes 48x48 pixel grids to isolate micro-expressions. By analyzing the subtle shifts in facial geometry, we identify your core emotional state with surgical precision.</p>
        </div>
        <div class="info-card card-2">
            <h4 style="color:#ff4b4b;">02 // BIOMETRIC SYNCING</h4>
            <p>Once your vibe is identified, our algorithm cross-references your emotion against a curated database of high-energy and soulful Bollywood tracks, ensuring the rhythm matches your internal frequency.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card card-3">
            <h4 style="color:#ff4b4b;">03 // HIGH-FIDELITY CURATION</h4>
            <p>Forget generic shuffling. Sentibeat AI connects you to specific playlists designed to amplify happiness, soothe sadness, or channel your raw energy through the power of Indian cinema's best music.</p>
        </div>
        <div class="info-card card-4">
            <h4 style="color:#ff4b4b;">04 // INSTANT PLAYBACK</h4>
            <p>The system bypasses traditional search menus, providing a direct link to your personalized audio experience. One scan, one click, total immersion in your emotional soundscape.</p>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE: LIVE & PHOTO SCANNER SHARED RESULT LOGIC ---
elif app_mode in ["LIVE SCANNER", "PHOTO ANALYSIS"]:
    st.markdown(f'<p class="brand-header">{app_mode}</p>', unsafe_allow_html=True)
    
    if app_mode == "LIVE SCANNER":
        source = st.camera_input("SCANNING...")
    else:
        source = st.file_uploader("UPLOAD SOURCE", type=['jpg', 'jpeg', 'png'])
        if source: st.image(source, width=300)

    if source:
        # Special Button for File Analysis
        if app_mode == "PHOTO ANALYSIS":
            if st.button("RUN DEEP ANALYSIS"):
                process_vibe(source, my_model, mood_labels, mood_playlists)
        else:
            # Auto-process for Camera
            process_vibe(source, my_model, mood_labels, mood_playlists)

    # DISPLAY RESULTS AND THE MUSIC BUTTON (Fixed for Music Opening)
    if st.session_state.detected_mood:
        st.markdown(f"""
            <div style="background-color: #111; padding: 30px; border-radius: 15px; border: 1px solid #333; margin-top:20px;">
                <h4 style="color: #666; letter-spacing: 5px;">RESULT_LOG</h4>
                <h1 style="color: #ff4b4b; font-size: 60px; margin: 0;">{st.session_state.detected_mood.upper()}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(f"▶️ LAUNCH {st.session_state.detected_mood.upper()} MIX", use_container_width=True):
            if st.session_state.detected_url:
                webbrowser.open_new_tab(st.session_state.detected_url)
                st.balloons()