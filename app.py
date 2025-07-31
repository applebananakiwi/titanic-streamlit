import streamlit as st
import joblib
import numpy as np

# Load your saved model
model = joblib.load("titanic_model.pkl")

# Title
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Number of Parents/Children", 0, 10, 0)
fare = st.number_input("Fare ($)", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Survived!")
    else:
        st.error("❌ Did Not Survive")
