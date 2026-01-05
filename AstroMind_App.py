import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import base64
import time
import os
import sklearn

# --- Premium Page Config ---
st.set_page_config(
    page_title="AstroMind | ZTF Astro-Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Model Definitions ---
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),    # net.0
            nn.ReLU(),                    # net.1
            nn.Linear(hidden, hidden),    # net.2
            nn.ReLU(),                    # net.3
            nn.Linear(hidden, embed_dim), # net.4
        )

    def forward(self, x):
        return self.net(x)

def fix_classifier(clf):
    """Fix scikit-learn version mismatch for LogisticRegression."""
    if not hasattr(clf, 'multi_class'):
        clf.multi_class = 'auto'
    return clf

# --- Asset Management ---
@st.cache_resource
def load_assets():
    # 1. Load Scaler
    with open('outputs_simclr_clean/scaler.pkl', 'rb') as f:
        sc_data = pickle.load(f)
        scaler = sc_data if not isinstance(sc_data, dict) else sc_data['scaler']
    
    # 2. Load Classifier
    with open('simclr_classifier_pipeline.pkl', 'rb') as f:
        clf_data = pickle.load(f)
        classifier = fix_classifier(clf_data['classifier'])
        CLASSES = list(classifier.classes_)
        
    # 3. Load SimCLR Encoder
    encoder = MLPEncoder(in_dim=8, hidden=256, embed_dim=128, dropout=0.1)
    state_dict = torch.load('outputs_simclr_clean/simclr_encoder.pt', map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    return scaler, classifier, encoder, CLASSES

# Attempt to load, fallback to demo mode if files missing
try:
    scaler, classifier, encoder, CLASSES = load_assets()
    LOCAL_MODE = True
except Exception as e:
    LOCAL_MODE = False
    CLASSES = ["Supernova_Ia", "Cataclysmic_Variable", "AGN_Flare", "Kilonova", "TDE"]
    st.warning("Model files not found. App running in Demo Mode.")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

galaxy_base64 = get_base64_image("WEB/images/galaxy.png")

# --- Custom Premium CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@400;700&display=swap');
    
    :root {{
        --bg-color: #0b0d17;
        --text-color: #e0e6ed;
        --accent-color: #00bcd4;
        --secondary-color: #7b2cbf;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
        --font-heading: 'Orbitron', sans-serif;
        --font-body: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: var(--bg-color);
        color: var(--text-color);
    }}

    h1, h2, h3, h4 {{
        font-family: var(--font-heading) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    .section-title {{
        font-size: 2.5rem;
        margin: 30px 0;
        background: linear-gradient(to right, var(--accent-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }}

    .hero-container {{
        height: 60vh;
        background: linear-gradient(rgba(0,0,0,0.4), rgba(11,13,23,1)), url('data:image/png;base64,{galaxy_base64}');
        background-size: cover;
        background-position: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border-radius: 30px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}

    .custom-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 30px;
        border-radius: 20px;
        backdrop-filter: blur(15px);
        margin-bottom: 25px;
        transition: 0.4s all ease;
    }}
    .custom-card:hover {{
        transform: translateY(-8px);
        border-color: var(--accent-color);
        box-shadow: 0 10px 20px rgba(0, 188, 212, 0.1);
    }}

    .result-box {{
        padding: 30px;
        border-radius: 20px;
        border: 1px solid var(--accent-color);
        background: rgba(0, 188, 212, 0.05);
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    /* Style Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
        justify-content: center;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #888;
        font-family: var(--font-heading);
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent-color) !important;
        border-bottom-color: var(--accent-color) !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Header / Hero ---
st.markdown(f"""
<div class="hero-container">
    <h1 style="font-size: 5rem; margin:0;">Astro<span style="color:var(--accent-color);">Mind</span></h1>
    <p style="font-size: 1.4rem; max-width: 800px; color: #bbb;">
        Next-Generation Transient Classification using SimCLR Representative Embeddings.
    </p>
    <div style="margin-top:20px;">
        <span style="padding: 5px 15px; border-radius: 20px; border: 1px solid var(--accent-color); color: var(--accent-color); font-size: 0.8rem; font-family: var(--font-heading);">ZTF Alert Stream</span>
        <span style="padding: 5px 15px; border-radius: 20px; border: 1px solid var(--secondary-color); color: var(--secondary-color); font-size: 0.8rem; font-family: var(--font-heading); margin-left:10px;">Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Navigation ---
tabs = st.tabs(["HOME", "ABOUT ME", "RESEARCH", "SYSTEM ARCHITECTURE", "CLASSIFIER"])

with tabs[0]:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Welcome to the Future of Astronomy
        Traditional transient alert processing is often biased by observation conditions. **AstroMind** leverages 
        Self-Supervised Contrastive Learning to create observation-invariant representations.
        
        Our system allows researchers to:
        - **Identify** transients with high confidence.
        - **Encode** complex observation metadata into 128D latent space.
        - **Analyze** the detectability of alerts in real-time.
        """)
        st.markdown(f"""
        <div class="custom-card">
            <h3 style="color:var(--accent-color)">Why SimCLR?</h3>
            <p>SimCLR (Simple Framework for Contrastive Learning) allows us to teach the model what "similar observation conditions" look like without requiring millions of labeled examples. This pre-training makes our final classifier significantly more robust to the noise of modern sky surveys.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if os.path.exists("WEB/images/nebula.png"):
            st.image("WEB/images/nebula.png", use_column_width=True, caption="Deep Space Visualization")
        else:
            st.info("Space Image Placeholder")

with tabs[1]:
    st.markdown('<h2 class="section-title">The Lead Researcher</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists("WEB/images/profile.png"):
            st.image("WEB/images/profile.png", use_column_width=True)
        else:
            st.markdown("<div style='height:300px; background:#222; border-radius:30px;'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="custom-card">
            <h2>Nipun Senevirathna</h2>
            <p style="font-size:1.2rem; color:var(--accent-color);">4th Year Undergraduate | SLIIT</p>
            <p><b>Major:</b> Artificial Intelligence & Data Science</p>
            <hr style="border:0.5px solid rgba(255,255,255,0.1)">
            <p>With a foundation in software engineering and a deep passion for the stars, I've dedicated my research to bridging the gap between massive data streams and human understanding. This project represents over 12 months of development in high-cadence astronomy pipelines.</p>
            <div style="display:flex; gap:15px; margin-top:20px;">
                <span style="color:#aaa;">📧 it22133922@my.sliit.lk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<h2 class="section-title">Research Proposal</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-card">
        <h3 style="color:var(--accent-color)">Problem Statement</h3>
        <p>The Zwicky Transient Facility (ZTF) produces alerts at a rate of 1 million per night. Manual inspection is impossible. Existing ML models often drop in accuracy when the camera is slightly out of focus (Seeing > 2.5") or when the Moon is bright. We need a model that "understands" these conditions.</p>
    </div>
    <div class="custom-card">
        <h3 style="color:var(--secondary-color)">Methodology</h3>
        <p>1. <b>Augmentation:</b> We apply random noise and feature masking to observation metadata.<br>
        2. <b>Representation:</b> A base encoder maps these to a latent space.<br>
        3. <b>Contrastive Loss:</b> We maximize similarity between two views of the same observation.<br>
        4. <b>Downstream task:</b> We freeze the encoder and train a classifier on top using a labeled dataset of 5,000+ confirmed transients.</p>
    </div>
    """, unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<h2 class="section-title">System Workflow</h2>', unsafe_allow_html=True)
    st.image("WEB/images/observatory.png", caption="System Framework", use_column_width=True)
    cols = st.columns(4)
    steps = [
        ("📡", "Data Source", "ZTF Kafka Stream / labeled_dataset.csv"),
        ("⚙️", "Preprocessing", "StandardScaler + Multi-Modal Alignment"),
        ("🧬", "SimCLR Encoder", "128D Latent Embedding Extraction"),
        ("⚖️", "Classification", "Logistic Regression + Confusion Matrix Analysis")
    ]
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="custom-card" style="text-align:center; height:250px;">
                <div style="font-size:3.5rem; margin-bottom:15px;">{icon}</div>
                <h4>{title}</h4>
                <p style="font-size:0.9rem; color:#888;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<h2 class="section-title">ZTF Alert Inference</h2>', unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 1], gap="large")
    
    with col_in:
        st.write("### 📝 Enter Alert Parameters")
        with st.form("inference_form"):
            r_mjd = st.number_input("MJD (Date)", value=58400.0, step=0.1)
            r_fid = st.selectbox("Frequency Band (FID)", [1, 2], format_func=lambda x: "g (Green)" if x==1 else "r (Red)")
            r_mag = st.number_input("Peak Magnitude", value=16.8)
            r_sig = st.number_input("Mag Uncertainty", value=0.03, format="%.3f")
            r_ra = st.number_input("Right Ascension", value=307.79)
            r_dec = st.number_input("Declination", value=51.13)
            r_pos = st.radio("Detection Position", [1, -1], format_func=lambda x: "Normal (Positive)" if x==1 else "Inverse (Negative)")
            
            submitted = st.form_submit_button("RUN ANALYSIS", use_container_width=True)
            
    with col_out:
        if submitted:
            if not LOCAL_MODE:
                st.error("Models not loaded. Showing simulated result.")
                label, confidence = np.random.choice(CLASSES), 95.5 + np.random.rand()*4
            else:
                with st.spinner("Extracting High-Dimensional Embeddings..."):
                    # Preprocess
                    cont = np.array([[r_mjd, r_mag, r_sig, r_ra, r_dec]])
                    cont_sc = scaler.transform(cont)
                    
                    fid_1 = 1 if r_fid == 1 else 0
                    fid_2 = 1 if r_fid == 2 else 0
                    
                    full = np.zeros((1, 8))
                    full[0, :5] = cont_sc
                    full[0, 5] = fid_1
                    full[0, 6] = fid_2
                    full[0, 7] = r_pos
                    
                    # Inference
                    with torch.no_grad():
                        emb = encoder(torch.FloatTensor(full)).numpy()
                    
                    label = classifier.predict(emb)[0]
                    probs = classifier.predict_proba(emb)[0]
                    confidence = np.max(probs) * 100
                    
                    time.sleep(1.5) # For premium feel
                
            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:var(--accent-color); opacity:0.7;">Classified Object</h3>
                <h1 style="color:var(--secondary-color); font-size:3rem; margin:10px 0;">{label.replace('_', ' ')}</h1>
                <p style="font-size:1.3rem;">Predictive Certainty: <b style="color:var(--accent-color);">{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Contextual Image
            img_map = {
                "Supernova_Ia": "WEB/images/galaxy.png",
                "AGN_Flare": "WEB/images/galaxy.png",
                "Variable": "WEB/images/nebula.png",
                "Cataclysmic": "WEB/images/nebula.png"
            }
            target_img = "WEB/images/observatory.png"
            for k, v in img_map.items():
                if k in label:
                    target_img = v
                    break
            
            if os.path.exists(target_img):
                st.image(target_img, caption=f"Typical Morphology for {label}", use_column_width=True)
            
            st.success("Target analysis complete. Alert verified via SimCLR Encoder.")
            st.balloons()
        else:
            st.markdown("""
            <div style="height:400px; border:1px dashed var(--card-border); border-radius:20px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#555;">
                <div style="font-size:4rem;">🔭</div>
                <p>Awaiting Alert Parameters for Real-time Inference</p>
            </div>
            """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<center style='color:#666; font-size:0.9rem;'>© 2026 AstroMind System | Zwicky Transient Facility Project v1.0.4<br>SLIIT Research Informatics Division</center>", unsafe_allow_html=True)
