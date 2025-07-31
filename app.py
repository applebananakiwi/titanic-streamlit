import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Header image
st.image("https://i.imgur.com/KZsmUi2.png", use_container_width=True)

# Title and description
st.title("🌊 Titanic: Will You Survive?")
st.markdown(
    """
    Welcome to the **Titanic Survival Predictor**!  
    Fill in the passenger details below and click **Predict** to see if they would have survived.
    """
)

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)

with col2:
    parch = st.number_input("Number of Parents/Children", 0, 10, 0)
    fare = st.number_input("Fare ($)", 0.0, 500.0, 50.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Predict button
st.markdown("---")
if st.button("🔹 Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("✅ Survived! 🎉")
        st.balloons()
    else:
        st.error("❌ Did Not Survive 😢")
        st.markdown(
            "<h3 style='text-align: center; color: gray;'>💐⚰️🌸 In Loving Memory 🌸⚰️💐</h3>",
            unsafe_allow_html=True
        )
        st.snow()  # Sad snowy effect
