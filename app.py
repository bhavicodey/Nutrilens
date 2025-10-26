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
from firebase_frontend_config import firebase_frontend_config

# ---------------------------
#  Streamlit page & theme
# ---------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
USDA_API_KEY = st.secrets.get("USDA_API_KEY") or os.getenv("USDA_API_KEY")
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
openai_key = os.getenv("OPENAI_API_KEY", "")
usda_key = os.getenv("USDA_API_KEY", "")
if not openai_key:
    st.error("Set OPENAI_API_KEY as environment variable to run this app.")
    st.stop()

client = OpenAI(api_key=openai_key)

# ---------------------------
#  Firebase setup
# ---------------------------
try:
    firebase = pyrebase.initialize_app(firebase_frontend_config)
    auth = firebase.auth()
    db = firebase.database()
    st.success("Firebase initialized successfully!")
except Exception as e:
    st.error(f"⚠️ Firebase initialization failed: {e}")
    db = None
    auth = None

# Only create firestore client if Firebase initialized
if firebase_admin._apps:
    db = firestore.client()
else:
    db = None

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

# Sidebar
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
# ---------------------------
#  USDA lookup / nutrition helpers
# ---------------------------
def get_usda_nutrition(food_name):
    if not usda_key or not food_name:
        return None
    try:
        search_term = food_name.lower().strip()
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={search_term}&pageSize=5&api_key={usda_key}"
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
        nutrients = {
            n["nutrientName"]: n["value"]
            for n in best_food.get("foodNutrients", [])
            if "nutrientName" in n
        }
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

# ---------------------------
#  Health scoring
# ---------------------------
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
    elif sat_fat > 4:
        score -= 10; explanation_parts.append("High saturated fat")
    if fiber >= 3:
        score += 5; explanation_parts.append("Good fiber")
    elif fiber < 1:
        score -= 10; explanation_parts.append("Low fiber")
    if protein >= 5:
        score += 5; explanation_parts.append("Good protein")
    score = max(0, min(100, score))
    color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
    explanation = ", ".join(explanation_parts) or "Balanced nutrition"
    return {"score": score, "color": color, "explanation": explanation}

# ---------------------------
#  Firebase save / fetch helpers
# ---------------------------
def save_to_firebase(user: str, food: str, score: int, nutrition: dict, confidence: float, serving_g: int, feedback=None):
    try:
        doc = {
            "user": user,
            "food": food,
            "score": int(score),
            "nutrition": nutrition,
            "confidence": float(confidence),
            "serving_g": int(serving_g),
            "feedback": feedback or None,
            "timestamp": datetime.utcnow().isoformat()
        }
        db.collection("food_results").add(doc)
        st.success("✅ Result saved to Firebase!")
    except Exception as e:
        st.warning(f"⚠️ Could not save to Firebase: {e}")

def fetch_user_history(user):
    try:
        docs = db.collection("food_results").where("user", "==", user).stream()
        rows = []
        for d in docs:
            data = d.to_dict()
            ts = data.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            rows.append({"date": dt.date(), "score": data.get("score", 0)})
        if not rows:
            return pd.DataFrame(columns=["date", "score"])
        df = pd.DataFrame(rows)
        agg = df.groupby("date").mean().reset_index().sort_values("date")
        return agg
    except Exception as e:
        st.warning(f"Could not fetch history: {e}")
        return pd.DataFrame(columns=["date", "score"])

def show_leaderboard():
    st.subheader("🏅 Leaderboard")
    try:
        users = {}
        docs = db.collection("food_results").stream()
        for d in docs:
            data = d.to_dict()
            u = data.get("user", "unknown")
            users.setdefault(u, {"count": 0, "total_score": 0})
            users[u]["count"] += 1
            users[u]["total_score"] += data.get("score", 0)
        leaderboard = [
            (u, v["count"], round(v["total_score"]/v["count"],1))
            for u,v in users.items() if v["count"]>0
        ]
        leaderboard.sort(key=lambda x:(-x[2],-x[1]))
        df=pd.DataFrame({
            "User":[u for u,_,_ in leaderboard],
            "Scans":[c for _,c,_ in leaderboard],
            "Avg Health Score":[s for _,_,s in leaderboard],
        })
        st.dataframe(df)
    except Exception as e:
        st.warning(f"Could not load leaderboard: {e}")

# ---------------------------
#  Ingredient explanations
# ---------------------------
def explain_flagged_ingredients(flagged_list):
    if not flagged_list:
        return ""
    prompt = (
        "You are a helpful nutrition educator. "
        f"A food label contains these ingredients: {', '.join(flagged_list)}. "
        "Write a concise, plain-language paragraph (3-5 sentences) explaining "
        "why each listed ingredient might be harmful or a concern for health. "
        "Mention common risks like excess sugar, preservatives, or artificial additives."
    )
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=250)
        text_out = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        text_out += getattr(c, "text", "")
        return text_out.strip()
    except Exception as e:
        st.warning(f"Could not generate ingredient explanation: {e}")
        return ""

# NEW FUNCTION — Explain safe/non-flagged ingredients
def explain_non_flagged_ingredients(non_flagged_list):
    if not non_flagged_list:
        return ""
    prompt = (
        "You are a nutrition educator. "
        f"The following ingredients appear safe or non-flagged: {', '.join(non_flagged_list)}. "
        "Write a concise paragraph (3-5 sentences) explaining why these ingredients are typically "
        "considered safe or beneficial in moderation. Mention if they provide nutrients, fiber, natural sources, "
        "or are common base ingredients. Keep it under 120 words."
    )
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=250)
        text_out = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        text_out += getattr(c, "text", "")
        return text_out.strip()
    except Exception as e:
        st.warning(f"Could not generate explanation for non-flagged ingredients: {e}")
        return ""

# ---------------------------
#  Main Tabs Setup
# ---------------------------
tab_scan, tab_stats, tab_leaderboard, tab_ingredients = st.tabs(
    ["🏠 Food Scan", "📈 Your Trends", "🏆 Leaderboard", "⚠️ Ingredient Scanner"]
)



# --------------------------
#  SCAN TAB
# --------------------------
#  SCAN TAB
# --------------------------
with tab_scan:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📸 Provide a food image or type the food name")
    st.markdown(
        "Upload, photograph, or manually type a food name. "
        "NutriLens will identify it, fetch nutrition, compute a health score, and suggest swaps.",
        unsafe_allow_html=True,
    )

    option = st.radio(
        "Input method:",
        ["Take a photo", "Upload from device", "Type food name"],
        horizontal=True,
        key="ingredient_source"
    )

    image = None
    food_text = None

    if option == "Take a photo":
        captured = st.camera_input("📷 Take a photo")
        if captured:
            image = Image.open(captured).convert("RGB")
            st.session_state["last_image"] = captured

    elif option == "Upload from device":
        uploaded = st.file_uploader("Upload food image", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state["last_image"] = uploaded

    else:  # Manual entry mode
        food_text = st.text_input("🍽️ Enter food name", placeholder="e.g., grilled chicken sandwich")

    # reuse last image if exists
    if image is None and "last_image" in st.session_state and not food_text:
        try:
            image = Image.open(st.session_state["last_image"]).convert("RGB")
        except Exception:
            image = None

    if image is None and not food_text:
        st.info("No input yet. Upload, photograph, or type a food name to begin.")
    else:
        # Identify label
        if food_text:
            label_raw = food_text
            confidence = 100.0
            st.success(f"Manually entered: **{label_raw}**")
        else:
            st.image(image, use_column_width=True, caption="Selected image")
            st.info("Classifying image locally...")
            label_raw, confidence, _ = classify_image(image)
            st.success(f"Model guess: **{label_raw}** — confidence **{confidence:.1f}%**")

        serving_g = st.slider("Estimate portion size (grams)", 25, 800, 100, 25)
        st.info("Refining label & nutrition with AI and USDA (if available)...")

        prompt = f"""
The identified food is: {label_raw}.
Clarify the food name (e.g., 'Lays Classic Potato Chips') 
and provide JSON:
- label
- nutrition: {{ calories, protein_g, fat_g, sugar_g, fiber_g, sodium_mg }}
- swap: healthier alternative
Respond in valid JSON only.
"""

        try:
            resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=400)
            text_out = ""
            for item in resp.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if getattr(c, "type", "") == "output_text":
                            text_out += getattr(c, "text", "")
            m = re.search(r"(\{.*\})", text_out, re.S)
            result = json.loads(m.group(1)) if m else {"label": label_raw, "nutrition": {}, "swap": "N/A"}

            usda = get_usda_nutrition(result.get("label", label_raw))
            if usda:
                scaled = {k: round(v * serving_g / 100.0, 2) for k, v in usda.items()}
                result["nutrition"] = scaled
                st.success("USDA data merged!")
            elif result.get("nutrition"):
                scaled = {k: round(v * serving_g / 100.0, 2) for k, v in result["nutrition"].items()}
                result["nutrition"] = scaled
                st.warning("Used AI-estimated nutrition (no USDA match).")

            health = compute_health_score(result.get("nutrition", {}), result.get("label", label_raw))
            result.update({
                "health_score": health["score"],
                "traffic_light": health["color"],
                "explanation": health["explanation"]
            })

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Food Identification")
                st.markdown(f"**{result.get('label','Unknown')}**")
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader(f"Nutrition Facts (~{serving_g} g)")
                if result.get("nutrition"):
                    nut = result["nutrition"]
                    nut_df = pd.DataFrame({
                        "Nutrient": ["Calories", "Protein (g)", "Total fat (g)", "Saturated fat (g)", "Carbohydrates (g)", "Sugar (g)", "Fiber (g)", "Sodium (mg)"],
                        "Amount": [
                            nut.get("calories", 0),
                            nut.get("protein_g", 0),
                            nut.get("fat_g", 0),
                            nut.get("saturated_fat_g", 0),
                            nut.get("carbohydrates_g", 0),
                            nut.get("sugar_g", 0),
                            nut.get("fiber_g", 0),
                            nut.get("sodium_mg", 0),
                        ]
                    })
                    st.table(nut_df.set_index("Nutrient"))
                else:
                    st.write("No nutrition data available.")
                st.markdown("**Healthy swap suggestion**")
                st.write(result.get("swap", "No suggestion"))
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='card metric'>", unsafe_allow_html=True)
                st.subheader("Health Score")
                st.markdown(f"<div class='traffic' style='font-size:28px'>{result['health_score']} — {result['traffic_light']}</div>", unsafe_allow_html=True)
                st.caption(result.get("explanation", ""))
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("Affordability")
                cost = 1.0
                st.metric("Est. cost / serving", f"${cost}")
                st.caption("Approximate per-serving cost")
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("Was this useful?")
            c1, c2 = st.columns(2)
            feedback = None
            if c1.button("👍 Correct"):
                feedback = "correct"
                st.success("Thanks — noted!")
            if c2.button("👎 Incorrect"):
                feedback = "incorrect"
                st.info("Thanks — we'll use this feedback.")

            user = st.session_state["user"]["email"] if "user" in st.session_state else "guest"
            save_to_firebase(user, result.get("label", label_raw), result.get("health_score", 0), result.get("nutrition", {}), confidence, serving_g, feedback)

            stats = fetch_user_history(user)
            if not stats.empty:
                st.subheader("Your Progress 🌱")
                st.line_chart(stats.set_index("date")["score"])

        except Exception as e:
            st.error(f"AI request failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------
#  INGREDIENT SCANNER TAB
# --------------------------
with tab_ingredients:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚠️ Ingredient & Nutrition Label Scanner")
    st.markdown("Upload or snap a **Nutrition Facts / Ingredients** label. The scanner will extract ingredients, flag harmful additives, and explain both flagged and safe ingredients.")

    opt = st.radio("Image source:", ["Take a photo", "Upload from device"], horizontal=True, key="label_source")
    image = None
    if opt == "Take a photo":
        captured = st.camera_input("Take a photo of the ingredient label")
        if captured:
            image = Image.open(captured).convert("RGB")
    else:
        up = st.file_uploader("Upload label image", type=["png", "jpg", "jpeg"])
        if up:
            image = Image.open(up).convert("RGB")

    if image is None:
        st.info("No label image yet. Upload or take a photo to extract ingredients.")
    else:
        st.image(image, caption="Uploaded Label", use_column_width=True)
        st.info("Analyzing label (vision + language model)...")

        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_b64}"

            prompt = """
            Extract all ingredients from this image.
            Identify harmful ones (for example: Red 40, Blue 1, Yellow 5/6, High Fructose Corn Syrup,
            Corn Syrup, MSG, Aspartame, Saccharin, Sodium Nitrate/Nitrite, BHT, BHA,
            Partially Hydrogenated Oils, Trans Fat, Artificial Flavors, Artificial Colors).
            Return valid JSON like:
            {
              "ingredients": [list],
              "flagged": [list of harmful items],
              "summary": "one-line summary for user"
            }
            """

            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": img_url},
                        ],
                    }
                ],
                max_output_tokens=500,
            )

            text_output = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if getattr(c, "type", "") == "output_text":
                            text_output += getattr(c, "text", "")

            m = re.search(r"(\{.*\})", text_output, re.S)
            if m:
                result = json.loads(m.group(1))
            else:
                result = {"ingredients": [], "flagged": [], "summary": "Could not parse response."}

            # Display extracted ingredients
            st.subheader("Extracted Ingredients")
            st.write(", ".join(result.get("ingredients", [])) or "No ingredients detected.")

            flagged = result.get("flagged", [])
            if flagged:
                st.markdown("<div class='flagged'>", unsafe_allow_html=True)
                st.error(f"⚠️ Harmful ingredients detected: {', '.join(flagged)}")
                st.markdown("</div>", unsafe_allow_html=True)

                explanation_paragraph = explain_flagged_ingredients(flagged)
                if explanation_paragraph:
                    st.subheader("Non-Approved Ingredients")
                    st.write(explanation_paragraph)
                else:
                    st.write("Could not generate an explanation for flagged ingredients.")

            else:
                st.markdown("<div class='healthy'>", unsafe_allow_html=True)
                st.success("✅ No harmful ingredients detected!")
                st.markdown("</div>", unsafe_allow_html=True)

            # NEW: Explain non-flagged ingredients
            all_ingredients = result.get("ingredients", [])
            non_flagged = [i for i in all_ingredients if i not in flagged]
            if non_flagged:
                non_flagged_explanation = explain_non_flagged_ingredients(non_flagged)
                if non_flagged_explanation:
                    st.subheader("Approved Ingredients")
                    st.write(non_flagged_explanation)

            st.caption(result.get("summary", ""))
            with st.expander("🔍 Full Model Output"):
                st.json(result)

        except Exception as e:
            st.error(f"Vision analysis failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
#  Footer / credits
# ---------------------------
st.write("---")
st.markdown("Built with ❤️  •  Model-powered insights. Data from USDA where available. These are estimates — consult a dietitian for medical advice.", unsafe_allow_html=True)
