from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Enter customer information to predict whether the customer is likely "
    "to purchase the Wellness Tourism Package."
)


# Path of the trained model saved by train.py / GitHub Actions
MODEL_PATH = Path("tourism_project/deployment/best_model.joblib")


@st.cache_resource
def load_model():
    """Load the trained model from the GitHub repository."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# Collect customer details
with st.form("customer_prediction_form"):

    st.subheader("Customer Details")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    typeof_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0.0,
        max_value=120.0,
        value=15.0,
        step=1.0
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Unmarried"]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    passport = st.selectbox(
        "Has Passport",
        [0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    pitch_satisfaction_score = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    own_car = st.selectbox(
        "Owns Car",
        [0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    predict_button = st.form_submit_button(
        "Predict Purchase Likelihood"
    )


# Create DataFrame and predict
if predict_button:

    input_data = pd.DataFrame(
        [{
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": number_of_person_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income
        }]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "Likely purchaser: the customer is predicted to purchase "
            "the Wellness Tourism Package."
        )
    else:
        st.warning(
            "Less likely purchaser: the customer is predicted not to purchase "
            "the Wellness Tourism Package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )

    with st.expander("View Customer Input DataFrame"):
        st.dataframe(input_data, use_container_width=True)
