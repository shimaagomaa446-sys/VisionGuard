import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
import cv2
import av
import sys
import json
import os
sys.path.append('.')

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from src.compliance import check_compliance

st.set_page_config(
    page_title="VisionGuard — AI Safety Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    html, body, [class*="css"] { color: #111827 !important; }
    .stApp { background-color: #FFFFFF !important; }
    
    .stApp p, .stApp h1, .stApp h2, .stApp h3, 
    .stApp h4, .stApp h5, .stApp h6, .stApp span,
    .stApp label, .stApp li, .stApp strong, .stApp b,
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #111827 !important;
    }
    
    /* HEADER */
    .header-section {
        padding: 3rem 0 2rem 0;
        text-align: center;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    
    .brand-logo {
        display: inline-flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    
    .brand-icon {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #DC2626 0%, #F59E0B 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.25);
    }
    
    .brand-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -1px;
        margin: 0;
    }
    
    .brand-subtitle {
        font-size: 0.95rem;
        color: #6B7280;
        font-weight: 500;
        letter-spacing: 0.3px;
        margin-top: 0.5rem;
    }
    
    .brand-badge {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.9rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    /* SIDEBAR - LIGHT */
    [data-testid="stSidebar"] {
        background: #F9FAFB !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.2rem;
    }
    
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] b,
    [data-testid="stSidebar"] div {
        color: #111827 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #374151 !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0 !important;
        border: none !important;
        border-top: 1px solid #E5E7EB !important;
    }
    
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider label p {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #4B5563 !important;
        font-size: 0.85rem !important;
        line-height: 1.6 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown b {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #F3F4F6 !important;
        border-color: #9CA3AF !important;
        transform: none !important;
    }
    
    [data-testid="stSidebar"] .stButton > button p {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.5rem 0.9rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #059669 !important;
        width: 100%;
        justify-content: center;
    }
    
    .status-badge.offline {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #DC2626 !important;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulseDot 2s infinite;
    }
    
    .status-online {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    
    .status-offline {
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
    }
    
    @keyframes pulseDot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid #E5E7EB;
        border-radius: 0;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 0.9rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: #6B7280 !important;
        background: transparent !important;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #111827 !important;
        background: #F9FAFB !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #DC2626 !important;
        background: transparent !important;
        border-bottom: 2px solid #DC2626 !important;
        box-shadow: none !important;
    }
    
    /* SECTION HEADERS */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #E5E7EB;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        letter-spacing: -0.3px;
    }
    
    /* METRIC CARDS */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.8rem 1.5rem;
        text-align: left;
        transition: all 0.25s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }
    
    .metric-card:hover {
        border-color: #D1D5DB;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    
    .metric-total { border-top: 4px solid #3B82F6; }
    .metric-compliant { border-top: 4px solid #10B981; }
    .metric-non-compliant { border-top: 4px solid #EF4444; }
    
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }
    
    .metric-icon { font-size: 1.5rem; opacity: 0.9; }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        color: #111827;
        letter-spacing: -2px;
        margin-top: 0.5rem;
    }
    
    .metric-value-total { color: #3B82F6; }
    .metric-value-compliant { color: #10B981; }
    .metric-value-non-compliant { color: #EF4444; }
    
    /* Mini metric for live */
    .mini-metric {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    
    .mini-metric-label {
        font-size: 0.7rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    
    .mini-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }
    
    /* PERSON CARDS */
    .person-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
        border-left: 4px solid;
    }
    
    .person-card:hover {
        border-color: #D1D5DB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .person-card-compliant {
        border-left-color: #10B981;
        background: linear-gradient(90deg, #F0FDF4 0%, #FFFFFF 8%);
    }
    
    .person-card-non-compliant {
        border-left-color: #EF4444;
        background: linear-gradient(90deg, #FEF2F2 0%, #FFFFFF 8%);
    }
    
    .person-title {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.2px;
    }
    
    .person-meta {
        font-size: 0.82rem;
        color: #6B7280;
        margin: 0.2rem 0;
        font-weight: 500;
    }
    
    .person-meta b { color: #374151; font-weight: 700; }
    
    /* BADGES */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-compliant {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .badge-violation {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: #111827 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        letter-spacing: 0.2px !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        background: #1F2937 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #F9FAFB !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 2px dashed #D1D5DB !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #DC2626 !important;
        background-color: #FEF2F2 !important;
    }
    
    [data-testid="stFileUploader"] * { color: #374151 !important; }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #6B7280 !important;
        font-size: 0.8rem !important;
    }
    
    /* ALERTS */
    .stAlert {
        border-radius: 8px !important;
        border-left-width: 4px !important;
        font-size: 0.9rem !important;
    }
    
    .stAlert p { color: inherit !important; }
    
    /* IMAGES */
    .stImage img {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    
    /* FOOTER */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #9CA3AF;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 3rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* TOAST */
    .custom-toast-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(4px);
        z-index: 9998;
    }
    
    .custom-toast {
        position: fixed;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: #FFFFFF;
        color: #111827;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
        animation: toastPop 0.4s ease-out;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.8rem;
        min-width: 300px;
        border: 1px solid #E5E7EB;
    }
    
    @keyframes toastPop {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
        100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }
    
    .custom-toast-icon { font-size: 3rem; animation: pulse 1.5s infinite; }
    .custom-toast-text { font-size: 1.1rem; font-weight: 700; color: #111827; }
    .custom-toast-subtext { font-size: 0.85rem; color: #6B7280; font-weight: 500; }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load YOLO Model
# ============================================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

live_model = load_model()

# ============================================
# Shared Stats File
# ============================================
STATS_FILE = "live_stats.json"

def init_stats_file():
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w') as f:
            json.dump({
                "current_persons": 0,
                "current_compliant": 0,
                "current_non_compliant": 0,
                "total_frames": 0,
                "accumulated_persons": 0,
                "accumulated_compliant": 0,
                "accumulated_non_compliant": 0
            }, f)

def read_stats():
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {
            "current_persons": 0,
            "current_compliant": 0,
            "current_non_compliant": 0,
            "total_frames": 0,
            "accumulated_persons": 0,
            "accumulated_compliant": 0,
            "accumulated_non_compliant": 0
        }

def write_stats(stats):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except:
        pass

def reset_stats():
    write_stats({
        "current_persons": 0,
        "current_compliant": 0,
        "current_non_compliant": 0,
        "total_frames": 0,
        "accumulated_persons": 0,
        "accumulated_compliant": 0,
        "accumulated_non_compliant": 0
    })

init_stats_file()

# ============================================
# Video Processor
# ============================================
class VisionGuardLiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_stats = {
            "total_persons": 0,
            "compliant": 0,
            "non_compliant": 0
        }
        self.accumulated = {
            "persons": 0,
            "compliant": 0,
            "non_compliant": 0
        }
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        try:
            results = live_model(img, conf=0.25, verbose=False)
            compliance = check_compliance(results, live_model)
            
            self.current_stats["total_persons"] = len(compliance)
            self.current_stats["compliant"] = sum(1 for c in compliance if c['status'] == 'COMPLIANT')
            self.current_stats["non_compliant"] = self.current_stats["total_persons"] - self.current_stats["compliant"]
            
            self.accumulated["persons"] += self.current_stats["total_persons"]
            self.accumulated["compliant"] += self.current_stats["compliant"]
            self.accumulated["non_compliant"] += self.current_stats["non_compliant"]
            
            write_stats({
                "current_persons": self.current_stats["total_persons"],
                "current_compliant": self.current_stats["compliant"],
                "current_non_compliant": self.current_stats["non_compliant"],
                "total_frames": self.frame_count,
                "accumulated_persons": self.accumulated["persons"],
                "accumulated_compliant": self.accumulated["compliant"],
                "accumulated_non_compliant": self.accumulated["non_compliant"]
            })
            
            annotated = results[0].plot()
        except Exception as e:
            print(f"Error: {e}")
            annotated = img.copy()
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
        
        cv2.putText(annotated, "VisionGuard Live", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated, f"Persons: {self.current_stats['total_persons']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Compliant: {self.current_stats['compliant']}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Non-Compliant: {self.current_stats['non_compliant']}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ============================================
# Header
# ============================================
st.markdown("""
<div class="header-section">
    <div class="brand-badge">✨ AI-Powered Safety</div>
    <div class="brand-logo">
        <div class="brand-icon">🦺</div>
        <h1 class="brand-text">VisionGuard</h1>
    </div>
    <p class="brand-subtitle">Computer Vision System for Construction Site Safety Monitoring</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown("### Configuration")
    
    conf_threshold = st.slider(
        'Confidence Threshold',
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    
    try:
        response = requests.get(f"http://127.0.0.1:8000/docs", timeout=2)
        st.markdown("""
        <div class="status-badge">
            <span class="status-dot status-online"></span>
            API Connected
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <div class="status-badge offline">
            <span class="status-dot status-offline"></span>
            API Offline
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Session")
    
    if st.button("Clear All Results", use_container_width=True):
        st.session_state.clear()
        reset_stats()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Detection Classes")
    st.markdown("""
    **PPE Items:**
    - ⛑️ Hardhat
    - 🦺 Safety Vest
    
    **Violations:**
    - ❌ NO-Hardhat
    - ❌ NO-Safety Vest
    
    **Person:**
    - 👷 Worker
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.75rem; padding: 1rem 0;">
        <b style="color: #6B7280;">VisionGuard v1.0</b><br>
        © 2026 Shimaa Gomaa
    </div>
    """, unsafe_allow_html=True)

# ============================================
# API URL
# ============================================
API_URL = 'http://127.0.0.1:8000'

# ============================================
# Helper Functions
# ============================================
def show_centered_toast(message="Analysis Complete!", icon="✅", duration=2.5):
    toast_placeholder = st.empty()
    toast_placeholder.markdown(f"""
    <div class="custom-toast-overlay"></div>
    <div class="custom-toast">
        <div class="custom-toast-icon">{icon}</div>
        <div class="custom-toast-text">{message}</div>
        <div class="custom-toast-subtext">Processing complete</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(duration)
    toast_placeholder.empty()


def show_summary_metrics(summary):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-total">
            <div class="metric-header">
                <div class="metric-label">Total Persons</div>
                <div class="metric-icon">👥</div>
            </div>
            <div class="metric-value metric-value-total">{summary.get('total_persons', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-compliant">
            <div class="metric-header">
                <div class="metric-label">Compliant</div>
                <div class="metric-icon">✅</div>
            </div>
            <div class="metric-value metric-value-compliant">{summary.get('compliant', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-non-compliant">
            <div class="metric-header">
                <div class="metric-label">Non-Compliant</div>
                <div class="metric-icon">⚠️</div>
            </div>
            <div class="metric-value metric-value-non-compliant">{summary.get('non_compliant', 0)}</div>
        </div>
        """, unsafe_allow_html=True)


def show_video_summary_metrics(summary):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-total">
            <div class="metric-header">
                <div class="metric-label">Unique Persons</div>
                <div class="metric-icon">👥</div>
            </div>
            <div class="metric-value metric-value-total">{summary.get('unique_persons', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-compliant">
            <div class="metric-header">
                <div class="metric-label">Compliant</div>
                <div class="metric-icon">✅</div>
            </div>
            <div class="metric-value metric-value-compliant">{summary.get('compliant', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-non-compliant">
            <div class="metric-header">
                <div class="metric-label">Non-Compliant</div>
                <div class="metric-icon">⚠️</div>
            </div>
            <div class="metric-value metric-value-non-compliant">{summary.get('non_compliant', 0)}</div>
        </div>
        """, unsafe_allow_html=True)


def show_person_cards(persons):
    for p in persons:
        status = p.get('status')
        person_id = p.get('person_id')
        conf = p.get('conf', 0)
        
        if status == 'COMPLIANT':
            st.markdown(f"""
            <div class="person-card person-card-compliant">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <div class="person-title">Worker #{person_id}</div>
                        <div class="person-meta">Confidence: <b>{conf:.1%}</b></div>
                    </div>
                    <span class="badge badge-compliant">✓ Compliant</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            violations = p.get("violations", [])
            badges = "".join([f'<span class="badge badge-violation">⚠ {v}</span>' for v in violations])
            
            st.markdown(f"""
            <div class="person-card person-card-non-compliant">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <div class="person-title">Worker #{person_id}</div>
                        <div class="person-meta">Confidence: <b>{conf:.1%}</b></div>
                    </div>
                    <div>{badges}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def show_video_person_cards(persons):
    for p in persons:
        status = p.get('status')
        track_id = p.get('track_id')
        violations = p.get('violations', [])
        appearances = p.get('appearances', 0)
        avg_conf = p.get('avg_confidence', 0)
        image_base64 = p.get('image')
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if image_base64:
                try:
                    img_bytes = base64.b64decode(image_base64)
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, use_container_width=True)
                except Exception:
                    st.markdown("*No image*")
            else:
                st.markdown("*No image*")
        
        with col2:
            if status == 'COMPLIANT':
                badges = '<span class="badge badge-compliant">✓ Compliant</span>'
                card_class = "person-card-compliant"
            else:
                badges = "".join([f'<span class="badge badge-violation">⚠ {v}</span>' for v in violations])
                card_class = "person-card-non-compliant"
            
            st.markdown(f"""
            <div class="person-card {card_class}" style="min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div class="person-title">Worker #{track_id}</div>
                    <div>{badges}</div>
                </div>
                <div class="person-meta">Tracked Frames: <b>{appearances}</b></div>
                <div class="person-meta">Avg Confidence: <b>{avg_conf:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)


def show_results(data, image):
    st.markdown('<h3 class="section-header">🖼️ Detection Results</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 Original Image**")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("**🔍 Detection Result**")
        if 'annotated_image' in data:
            img_bytes = base64.b64decode(data['annotated_image'])
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, use_container_width=True)
    
    st.markdown('<h3 class="section-header">📊 Summary</h3>', unsafe_allow_html=True)
    show_summary_metrics(data.get('summary', {}))
    
    persons = data.get('results', [])
    if persons:
        st.markdown('<h3 class="section-header">👤 Persons Details</h3>', unsafe_allow_html=True)
        show_person_cards(persons)


# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image", "🎥 Video", "🎬 Live", "📖 About"])

# ============================================
# Tab 1: Image
# ============================================
with tab1:
    st.markdown('<h3 class="section-header">📸 Analyze Image</h3>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="img_uploader")
    
    if uploaded:
        if st.session_state.get('current_file') != uploaded.name:
            st.session_state.pop('results', None)
            st.session_state['current_file'] = uploaded.name
        
        if st.button("🚀 Analyze Image", use_container_width=True, key="btn_img"):
            with st.spinner("🔍 Analyzing..."):
                files = {'file': (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(f"{API_URL}/analyze", files=files)
                
                if response.status_code == 200:
                    st.session_state['results'] = response.json()
                    show_centered_toast("Image Analyzed Successfully!", "🎯")
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)
    
    if 'results' in st.session_state and uploaded:
        show_results(st.session_state['results'], uploaded)

# ============================================
# Tab 2: Video
# ============================================
with tab2:
    st.markdown('<h3 class="section-header">🎥 Analyze Video</h3>', unsafe_allow_html=True)
    st.info("⚠️ Video analysis can take 1-3 minutes. We use tracking + sampling (every 5 frames) for accuracy and speed.")
    
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov'],
        key="video_uploader"
    )
    
    if uploaded_video:
        if st.session_state.get('current_video') != uploaded_video.name:
            st.session_state.pop('video_results', None)
            st.session_state['current_video'] = uploaded_video.name
        
        st.markdown("**📹 Original Video**")
        st.video(uploaded_video)
        
        if st.button("🚀 Analyze Video", use_container_width=True, key="btn_video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎬 Analyzing video... This may take a few minutes."):
                status_text.text("📤 Uploading video...")
                progress_bar.progress(20)
                
                files = {'file': (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                
                status_text.text("🔍 Processing frames with tracking...")
                progress_bar.progress(50)
                
                response = requests.post(f"{API_URL}/analyze-video", files=files)
                
                progress_bar.progress(90)
                
                if response.status_code == 200:
                    st.session_state['video_results'] = response.json()
                    progress_bar.progress(100)
                    status_text.text("✅ Done!")
                    show_centered_toast("Video Analyzed Successfully!", "🎬", duration=3)
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)
    
    if 'video_results' in st.session_state:
        data = st.session_state['video_results']
        summary = data.get('summary', {})
        persons = data.get('persons', [])
        
        st.markdown('<h3 class="section-header">📊 Video Info</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("🎞️ Total Frames", summary.get('total_frames', 0))
        col2.metric("🔍 Analyzed Frames", summary.get('processed_frames', 0))
        col3.metric("⚡ FPS", summary.get('fps', 0))
        
        st.markdown('<h3 class="section-header">📊 Summary</h3>', unsafe_allow_html=True)
        show_video_summary_metrics(summary)
        
        if persons:
            st.markdown('<h3 class="section-header">👤 Persons Detected</h3>', unsafe_allow_html=True)
            show_video_person_cards(persons)

# ============================================
# Tab 3: Live Camera
# ============================================
with tab3:
    st.markdown('<h3 class="section-header">🎬 Live Camera (Real-time)</h3>', unsafe_allow_html=True)
    st.info("⚡ Real-time detection. Click **Refresh Summary** to update the panel.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_streamer(
            key="visionguard-live",
            video_processor_factory=VisionGuardLiveProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.markdown("### 🔴 Live Summary")
        
        if st.button("🔄 Refresh Summary", use_container_width=True, key="btn_refresh_live"):
            st.rerun()
        
        stats = read_stats()
        
        st.markdown("**Current Frame:**")
        
        st.markdown(f"""
        <div class="mini-metric" style="border-top: 3px solid #3B82F6;">
            <div class="mini-metric-label">👥 Persons</div>
            <div class="mini-metric-value" style="color: #3B82F6;">{stats['current_persons']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="mini-metric" style="border-top: 3px solid #10B981;">
            <div class="mini-metric-label">✅ Compliant</div>
            <div class="mini-metric-value" style="color: #10B981;">{stats['current_compliant']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="mini-metric" style="border-top: 3px solid #EF4444;">
            <div class="mini-metric-label">⚠️ Non-Compliant</div>
            <div class="mini-metric-value" style="color: #EF4444;">{stats['current_non_compliant']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Cumulative (Session):**")
        
        st.markdown(f"""
        <div class="mini-metric" style="border-top: 3px solid #6B7280;">
            <div class="mini-metric-label">🎞️ Frames Processed</div>
            <div class="mini-metric-value" style="color: #374151; font-size: 1.4rem;">{stats['total_frames']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="mini-metric" style="border-top: 3px solid #3B82F6;">
            <div class="mini-metric-label">👥 Total Detections</div>
            <div class="mini-metric-value" style="color: #3B82F6; font-size: 1.4rem;">{stats['accumulated_persons']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Click **START** on the camera
        2. Allow camera access
        3. Point at workers
        4. Click **Refresh Summary** for updates
        """)
        
        st.markdown("---")
        st.warning("""
        ⚠️ Live mode is approximate (~3-8 FPS).
        For accurate results, use **Image** or **Video** tabs.
        """)

# ============================================
# Tab 4: About
# ============================================
with tab4:
    st.markdown('<h3 class="section-header">📖 About VisionGuard</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    VisionGuard is a computer vision system that detects PPE (Personal 
    Protective Equipment) violations in construction sites.
    
    ### 🛠️ Technology Stack
    
    - **YOLOv8s** — Object detection model
    - **FastAPI** — Backend API
    - **Streamlit** — Frontend UI
    - **streamlit-webrtc** — Real-time video
    
    ### 📊 Model Performance
    
    - **mAP50:** 0.811
    - **mAP50-95:** 0.517
    - **Precision:** 0.903
    - **Recall:** 0.777
    
    ### ✨ Features
    
    - 📸 **Image Analysis** — Upload and analyze images
    - 🎥 **Video Analysis** — Upload videos with tracking
    - 🎬 **Live Camera** — Real-time detection & compliance
    
    ### 👨‍💻 Author
    
    Shimaa Gomaa
    """)
    
    st.markdown("---")
    st.markdown("""
    <div class="app-footer">
        <b>VisionGuard</b> · AI Safety Monitoring System · v1.0<br>
        © 2026 Shimaa Gomaa · Built with YOLOv8, FastAPI & Streamlit
    </div>
    """, unsafe_allow_html=True)