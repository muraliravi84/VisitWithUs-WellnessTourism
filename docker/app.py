
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face model repo
MODEL_REPO = "Murali0606/wellness_tourism_model"

# Load model from Hugging Face Hub (cached after first download)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="random_forest.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("VisitWithUs - Wellness Tourism Prediction")
st.write("Predict whether a customer will purchase a tourism package.")

# Input form
with st.form("customer_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=20, value=2)
    Passport = st.selectbox("Passport", [0, 1])
    OwnCar = st.selectbox("Own Car", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Premium"])
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=2)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

    #  Submit button must be inside the form block
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build dataframe with same schema as training
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "ProductPitched": ProductPitched,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch
    }])

  #  Drop accidental index column if present
    input_data = input_data.drop(columns=["Unnamed: 0"], errors="ignore")

    # Get probabilities
    proba = model.predict_proba(input_data)[0]
    prediction = (proba[1] > 0.5).astype(int)  # threshold at 50%

    st.write(f"Probability of purchase: {proba[1]:.2f}")
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("This customer is likely to purchase the tourism package!")
    else:
        st.warning("This customer is unlikely to purchase the tourism package.")


