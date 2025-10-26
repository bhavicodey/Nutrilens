import os
import streamlit as st
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from openai import OpenAI
import json
import re
from datetime import datetime
import pandas as pd
import io, base64

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

# ---------------------------
#  Streamlit page & theme
# ---------------------------
st.set_page_config(
    page_title="NutriLens (Food + Nutrition)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .card {background: var(--bg-card, #ffffff); padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); margin-bottom: 12px;}
    .small {font-size:0.9rem; color: #6b7280;}
    .metric {border-radius:12px; padding:10px;}
    .traffic {font-weight:700;}
    .flagged {background:#fff4f4; border-left:4px solid #ff4d4f; padding:12px; border-radius:8px;}
    .healthy {background:#f4fff4; border-left:4px solid #2ecc71; padding:12px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='margin-bottom:6px'>NutriLens — Snap food, get nutrition & health score 🍎📸</h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Identify foods, fetch USDA nutrition, compute a health score, and flag harmful ingredients.</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
#  API Keys & Clients
# ---------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
USDA_API_KEY = st.secrets.get("USDA_API_KEY") or os.getenv("USDA_API_KEY")
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY as a Streamlit secret to run this app.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
#  Firebase setup via secrets
# ---------------------------
if not firebase_admin._apps:
    firebase_creds = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()

firebase_frontend_config = {
    "apiKey": st.secrets["FIREBASE_API_KEY"],
    "authDomain": st.secrets["FIREBASE_AUTH_DOMAIN"],
    "projectId": st.secrets["FIREBASE_PROJECT_ID"],
    "storageBucket": st.secrets["FIREBASE_STORAGE_BUCKET"],
    "messagingSenderId": st.secrets["FIREBASE_MESSAGING_SENDER_ID"],
    "appId": st.secrets["FIREBASE_APP_ID"],
    "measurementId": st.secrets["FIREBASE_MEASUREMENT_ID"],
    "databaseURL": st.secrets["FIREBASE_DATABASE_URL"],
}

firebase = pyrebase.initialize_app(firebase_frontend_config)
auth = firebase.auth()

# ---------------------------
#  Authentication helpers
# ---------------------------
def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state["user"] = user
        st.success(f"Welcome back, {email}!")
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

def signup_user(email, password):
    try:
        auth.create_user_with_email_and_password(email, password)
        st.success("✅ Account created! Please log in.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

# ---------------------------
#  Sidebar
# ---------------------------
with st.sidebar:
    st.header("🔐 Account")
    if "user" not in st.session_state:
        auth_mode = st.radio("Choose an option", ["Login", "Sign up"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if auth_mode == "Login":
            if st.button("Login"):
                if login_user(email, password):
                    st.experimental_rerun()
        else:
            if st.button("Sign up"):
                signup_user(email, password)
    else:
        user_info = st.session_state["user"]
        email = user_info.get("email", "Unknown")
        st.success(f"Logged in as {email}")
        if st.button("Logout"):
            del st.session_state["user"]
            st.experimental_rerun()

    st.write("---")
    st.markdown("### Tips")
    st.markdown("- Upload a clear picture of the food or label.\n- Adjust the portion size slider for better estimates.\n- Save scans to build streaks & track progress.")
    st.write("---")
    if st.button("Clear cached model"):
        for k in list(st.session_state.keys()):
            if k.startswith("model_cached_"):
                del st.session_state[k]
        st.success("Cleared.")

# ---------------------------
#  Model loading
# ---------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(url).text.splitlines()
    return labels

model = load_model()
labels = load_labels()

# ---------------------------
#  Classification + Nutrition helpers
# ---------------------------
def classify_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, k=1)
        predicted_index = top_idx[0][0].item()
        confidence = top_prob[0][0].item() * 100.0
        label = labels[predicted_index]
    return label, confidence, probs

def get_usda_nutrition(food_name):
    if not USDA_API_KEY or not food_name:
        return None
    try:
        search_term = food_name.lower().strip()
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={search_term}&pageSize=5&api_key={USDA_API_KEY}"
        r = requests.get(url, timeout=8)
        data = r.json()
        foods = data.get("foods", [])
        if not foods:
            return None
        keywords = search_term.split()
        best_food = None
        for food in foods:
            desc = food.get("description", "").lower()
            if any(k in desc for k in keywords):
                best_food = food
                break
        if not best_food:
            best_food = foods[0]
        nutrients = {n["nutrientName"]: n["value"] for n in best_food.get("foodNutrients", []) if "nutrientName" in n}
        return {
            "calories": nutrients.get("Energy", 0),
            "protein_g": nutrients.get("Protein", 0),
            "fat_g": nutrients.get("Total lipid (fat)", 0),
            "saturated_fat_g": nutrients.get("Fatty acids, total saturated", 0),
            "carbohydrates_g": nutrients.get("Carbohydrate, by difference", 0),
            "sugar_g": nutrients.get("Sugars, total", 0),
            "fiber_g": nutrients.get("Fiber, total dietary", 0),
            "sodium_mg": nutrients.get("Sodium, Na", 0),
        }
    except Exception as e:
        st.warning(f"USDA lookup failed: {e}")
        return None

def compute_health_score(nutrition, label=""):
    if not nutrition:
        return {"score": 50, "color": "🟡", "explanation": "Incomplete data"}
    score = 100
    explanation_parts = []
    ultra_processed_keywords = ["chips", "cookie", "candy", "soda", "frito", "snack", "crackers", "pretzel"]
    if any(k in label.lower() for k in ultra_processed_keywords):
        score -= 40
        explanation_parts.append("Ultra-processed food")
    sugar = nutrition.get("sugar_g", 0)
    sodium = nutrition.get("sodium_mg", 0)
    fiber = nutrition.get("fiber_g", 0)
    sat_fat = nutrition.get("saturated_fat_g", 0)
    protein = nutrition.get("protein_g", 0)
    if sugar > 20:
        score -= 25; explanation_parts.append("High sugar")
    elif sugar > 10:
        score -= 10; explanation_parts.append("Moderate sugar")
    if sodium > 1000:
        score -= 30; explanation_parts.append("Very high sodium")
    elif sodium > 600:
        score -= 15; explanation_parts.append("High sodium")
    if sat_fat > 8:
        score -= 25; explanation_parts.append("Very high saturated fat")
