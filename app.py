import sys
print("Python executable:", sys.executable)
import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# --- CSS for Background & Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://muralsyourway.vtexassets.com/arquivos/ids/238964/Underwater-Ocean-Bottom-Mural-Wallpaper.jpg?v=638164497054230000");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    .main {
        background-color: rgba(0, 0, 0, 0.6);  /* darker transparent background */
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
        max-width: 700px;
        margin: auto;
    }

    h1, h2, h3, h4, h5, h6, label, .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
        color: white !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }

    html, body, [class*="css"] {
        color: white !important;
        font-size: 18px;
        font-weight: bold;
    }

    .stSlider > div > div {
        color: white !important;
        font-weight: bold;
    }

    .stNumberInput input, .stTextInput input {
        color: white;
        background-color: #222;
        font-weight: bold;
    }

    .stSelectbox div[data-baseweb="select"] {
        color: white;
        background-color: #222;
        font-weight: bold;
    }

    .stButton>button {
        color: white;
        background-color: #0073e6;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #005bb5;
        transform: scale(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🚢 Titanic Survival Prediction")

# --- Inputs ---
pclass = st.selectbox("🛳️ Passenger Class", [1, 2, 3])
sex = st.selectbox("👤 Sex", ["male", "female"])
age = st.slider("🎂 Age", 1, 80, 25)
sibsp = st.number_input("👨‍👩‍👧‍👦 Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("👪 Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("💵 Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("🚢 Port of Embarkation", ["S", "C", "Q"])

# --- Encoding ---
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"S": 2, "C": 0, "Q": 1}
embarked_encoded = embarked_map[embarked]

# --- Model Load & Prediction ---
model = joblib.load("titanic_model.pkl")
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
prediction = model.predict(input_data)

# --- Result ---
if st.button("Predict"):
    if prediction[0] == 1:
        st.success("🎉 You would have survived!")
    else:
        st.error("💀 Unfortunately, you would not have survived.")

st.markdown("</div>", unsafe_allow_html=True)
