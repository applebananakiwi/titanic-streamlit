import streamlit as st
import joblib
import numpy as np

# Load your saved model
model = joblib.load("titanic_model.pkl")

# Sidebar Styling
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Titanic_side_plan.png", use_column_width=True)
st.sidebar.header("🚢 Titanic Survival Prediction")

st.sidebar.markdown(
    """
    **Enter passenger details below to predict survival.**  
    _0 = Did Not Survive_  
    _1 = Survived_
    """
)

# Input fields in the sidebar
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children", 0, 10, 0)
fare = st.sidebar.number_input("Fare ($)", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Main page title and info
st.title("🌊 Titanic: Will You Survive?")
st.markdown(
    """
    Welcome to the **Titanic Survival Predictor**!  
    Fill in the passenger details in the **sidebar** and click **Predict** to see if they would have survived.
    """
)

# Prediction button
if st.sidebar.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)
    st.markdown("---")
    st.subheader("🔹 Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Survived!")
        st.balloons()  # Adds a nice animation
    else:
        st.error("❌ Did Not Survive")
