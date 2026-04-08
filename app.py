import streamlit as st
import pandas as pd
import scipy.sparse as sp
import pickle

# PAGE SETUP
st.set_page_config(page_title="RoleCast AI", page_icon="🎯", layout="centered")

st.title("🎯 RoleCast AI: Job Role Predictor")
st.markdown("Enter candidate details to predict the most suitable AI job role.")
st.divider()

# LOAD MODEL (FIXED)
@st.cache_resource
def load_model():
    with open('rolecast_rf_model.pkl', 'rb') as f:
        data = pickle.load(f)

    return (
        data['model'],
        data['count_vectorizer'],
        data['label_encoders'],
        data['target_le'],
        data['categorical_cols'],
        data['numeric_cols']
    )

rf_model, count_vectorizer, label_encoders, target_le, categorical_cols, numeric_cols = load_model()

# USER INPUT FORM
with st.form("prediction_form"):

    st.subheader("👤 Candidate Profile")

    col1, col2 = st.columns(2)

    with col1:
        experience = st.selectbox("Experience Level", label_encoders['experience_level'].classes_)
        education = st.selectbox("Education Level", label_encoders['education_level'].classes_)
        industry = st.selectbox("Industry", label_encoders['industry'].classes_)

    with col2:
        employment = st.selectbox("Employment Type", label_encoders['employment_type'].classes_)
        company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)
        remote = st.selectbox("Remote Friendly", label_encoders['remote_friendly'].classes_)

    years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
    salary = st.number_input("Expected Salary (USD)", min_value=0, value=50000)

    st.subheader("💻 Technical Skills")
    skills = st.text_area("Skills", "Python, Machine Learning, SQL")
    tools = st.text_area("Tools & Technologies", "TensorFlow, AWS")
    certs = st.text_area("Certifications", "none")

    submit_button = st.form_submit_button("🔮 Predict Job Role")

# PREDICTION
if submit_button:

    # Create dataframe
    user_data = {
        "experience_level": experience,
        "education_level": education,
        "industry": industry,
        "employment_type": employment,
        "company_size": company_size,
        "remote_friendly": remote,
        "years_of_experience": years_exp,
        "annual_salary_usd": salary
    }

    user_df = pd.DataFrame([user_data])

    # APPLY SAME LABEL ENCODERS (IMPORTANT)
    for col in categorical_cols:
        le = label_encoders[col]
        user_df[col] = le.transform(user_df[col])

    # Combine skills
    skills_combined = f"{skills} {tools} {certs}"
    user_skills_matrix = count_vectorizer.transform([skills_combined])

    # Combine features
    X_user = sp.hstack([sp.csr_matrix(user_df.values), user_skills_matrix])

    # Predict
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