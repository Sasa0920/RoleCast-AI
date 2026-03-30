import streamlit as st
import pandas as pd
import scipy.sparse as sp
import pickle

# PAGE SETUP
st.set_page_config(page_title="RoleCast AI", page_icon="🎯", layout="centered")

st.title("🎯 RoleCast AI: Job Role Predictor")
st.markdown("Enter candidate details to predict the most suitable AI job role.")
st.divider()

# LOAD MODELS
@st.cache_resource
def load_models():
    with open('models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/count_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('models/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)

    return model, vectorizer, target_encoder

rf_model, count_vectorizer, target_le = load_models()

# USER INPUT FORM
with st.form("prediction_form"):

    st.subheader("👤 Candidate Profile")

    col1, col2 = st.columns(2)

    with col1:
        experience = st.selectbox("Experience Level", ["Entry", "Mid", "Senior"])
        education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
        industry = st.selectbox("Industry", [
            "Tech","Finance","Healthcare","Education","Retail","E-commerce",
            "Defense","Insurance","Automotive","SaaS","Legal","Hedge Fund",
            "Robotics","Marketing","Media","Manufacturing","Banking",
            "Logistics","Consulting","Telecom","Academia"
        ])

    with col2:
        employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Remote"])
        company_size = st.selectbox("Company Size", ["Startup", "Mid-size", "Large"])
        remote = st.selectbox("Remote Friendly", ["Yes", "No", "Hybrid"])

    years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)

    st.subheader("💻 Technical Skills")
    skills = st.text_area("Skills", "Python, Machine Learning, SQL")
    tools = st.text_area("Tools & Technologies", "TensorFlow, AWS")

    submit_button = st.form_submit_button("🔮 Predict Job Role")

# ENCODING 
encoding_maps = {
    "experience_level": {"Entry": 0, "Mid": 1, "Senior": 2},
    "education_level": {"Bachelor's": 0, "Master's": 1, "PhD": 2},
    "employment_type": {"Contract": 0, "Full-time": 1, "Internship": 2, "Remote": 3},
    "company_size": {"Large": 0, "Mid-size": 1, "Startup": 2},
    "remote_friendly": {"Hybrid": 0, "No": 1, "Yes": 2}
}


# PREDICTION
if submit_button:

    user_data = {
        "experience_level": encoding_maps["experience_level"][experience],
        "education_level": encoding_maps["education_level"][education],
        "industry": industry,  # will encode below
        "employment_type": encoding_maps["employment_type"].get(employment, 0),
        "company_size": encoding_maps["company_size"][company_size],
        "remote_friendly": encoding_maps["remote_friendly"][remote],
        "years_of_experience": years_exp
    }

    user_df = pd.DataFrame([user_data])

   
    try:
        industry_encoded = list(target_le.classes_).index(industry)
    except:
        industry_encoded = 0

    user_df["industry"] = industry_encoded

    skills_combined = f"{skills} {tools}"
    user_skills_matrix = count_vectorizer.transform([skills_combined])

    X_user = sp.hstack([sp.csr_matrix(user_df.values), user_skills_matrix])

    probabilities = rf_model.predict_proba(X_user)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]

    st.divider()
    st.header("🏆 Prediction Result")

    best_role = target_le.inverse_transform([top_3_indices[0]])[0]
    best_score = probabilities[top_3_indices[0]] * 100

    st.success(f"🎯 Best Match: {best_role} ({best_score:.2f}%)")

    st.write("### 🔄 Other Possible Roles:")
    for idx in top_3_indices[1:]:
        role = target_le.inverse_transform([idx])[0]
        score = probabilities[idx] * 100
        st.progress(int(score), text=f"{role} — {score:.2f}%")