import sys
print("Python executable:", sys.executable)
import streamlit as st
import joblib
import numpy as np

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode categorical values
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"S": 2, "C": 0, "Q": 1}
embarked_encoded = embarked_map[embarked]

# Load trained model
model = joblib.load("titanic_model.pkl")

# Make prediction
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
prediction = model.predict(input_data)

if st.button("Predict"):
    if prediction[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("❌ Did not survive.")
