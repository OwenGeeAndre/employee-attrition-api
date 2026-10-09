# 1. Importing tools

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# 2. Loading prediction model

@st.cache_resource     # loads the model once
def load_model():
    return joblib.load("hr_employee_attrition.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️⚠️Model file not found: ensure 'hr_employee_attrition.pkl' is in the same folder")
    st.stop()

# 3. webpage setup & title

st.set_page_config(page_title="Employee Attrition Risk Predictor", page_icon="🏢", layout="wide")

st.title("HR: Employee Attrition & Flight Risk Predictor")
st.write("Adjust the employee profile below to calculate the likelihood of them leaving the company")

st.markdown("---")

# 4. inputs (grouped by meta data)
# NOTE: we don't include the columns that we dropped before training model

c1, c2, c3 = st.columns(3)    # creating 3 columns

# COLUMN 1: work and role details
with c1:
    st.subheader("Role and Work life 👨‍🔧")

    Department = st.selectbox(
        "Department",
        ['Sales', 'Research & Development', 'Human Resources']
    )

    JobRole = st.selectbox(
        "Job Role",
        ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
         'Manufacturing Director', 'Healthcare Representative', 'Manager',
         'Sales Representative', 'Research Director', 'Human Resources']
    )

    BusinessTravel = st.selectbox(
        "Business Travel Frequency",
        ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    )

    OverTime = st.radio("Works overtime?", ['No','Yes'], horizontal=True)

    DistanceFromHome = st.slider('Distance from home (miles)', min_value=1, max_value=30, value=8)


# COLUMN 2: experience and compensation

with c2:
    st.subheader("Demographics and Tenure 📈")

    AgeGroup = st.selectbox(
        "Age Group",
        ['35-44', '45-54', '25-34', '55+', 'Under 25'],
        index = 4 # default to Under 25
    )

    TenureGroup = st.selectbox(
        "Period at Company",
        ['Established (6-10 yrs)', 'New (0-2 yrs)', 'Early (3-5 yrs)', 'Veteran (10+ yrs)'],
        index = 1
    )

    MonthlyIncomeBracket = st.selectbox(
        "Monthly Income Range",
        ['High', 'Low', 'Medium', 'Very High'],
        index = 1
    )

    Gender = st.radio("Gender", ['Female', 'Male'], horizontal=True)

    MaritalStatus = st.selectbox(
        "Marital Status",
        ['Single', 'Married', 'Divorced']
    )

    EducationField = st.selectbox(
        "Field of Study",
        ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
    )

# COLUMN 3: satisfaction ratings

with c3:
    st.subheader("Employee Sentiment (1 to 4)")

    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    JobInvolvement = st.slider("Job Involvement", 1, 4, 3)
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    StockOptionLevel = st.selectbox("Stock Option Level", [0,1,2,3], index=1)
    WorkLifeBalance = st.slider("Work-Life Balance", 1, 4, 3)

    
# Other details (collapsible to save space and keep page clean)

with st.expander("Additional Work History"):
    sub1, sub2, sub3 = st.columns(3)
    with sub1: 
        Education = st.selectbox("Education Level (1=Below College, 2=College, 3=Bachelors, 4=Masters, 5=Doctorate)",
                                [1,2,3,4,5], index=2)
        HourlyRate = st.number_input("Hourly Rate ($)", value=65)
    with sub2:
        NumCompaniesWorked = st.slider("Previous Companies Worked", 0, 9, 2)
        PercentSalaryHike = st.slider("Percent Salary Hike (%)", 0, 30, 10)
        PerformanceRating = st.selectbox("Performance Rating", [0, 1, 2, 3, 4], index=3)
    with sub3:
        TrainingTimesLastYear = st.slider("Trainings Attended Last Year", 0, 6, 2)
        YearsInCurrentRole = st.slider("Years in Current Role", 0, 10, 2)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 20, 1)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, 3)

# calculate overall satisfaction index
OverallSatisfactionIndex = (EnvironmentSatisfaction + JobSatisfaction + RelationshipSatisfaction + WorkLifeBalance) / 4.0
        
# 5. prediction pipeline

st.markdown("---")

if st.button("Calculate Attrition Risk", type="primary"):
    
    # the model needs all input, let's create a 1-row DataFrame of inputs
    input_data = pd.DataFrame(
        [
            {
                'BusinessTravel':BusinessTravel,
                'Department':Department,
                'DistanceFromHome':DistanceFromHome,
                'Education':Education,
                'EducationField':EducationField,
                'EnvironmentSatisfaction':EnvironmentSatisfaction,
                'Gender':Gender,
                'HourlyRate':HourlyRate,
                'JobInvolvement':JobInvolvement,
                'JobRole':JobRole,
                'JobSatisfaction':JobSatisfaction,
                'MaritalStatus':MaritalStatus,
                'NumCompaniesWorked':NumCompaniesWorked,
                'OverTime':OverTime,
                'PercentSalaryHike':PercentSalaryHike,
                'PerformanceRating':PerformanceRating,
                'RelationshipSatisfaction':RelationshipSatisfaction,
                'StockOptionLevel':StockOptionLevel,
                'TrainingTimesLastYear':TrainingTimesLastYear,
                'WorkLifeBalance':WorkLifeBalance,
                'YearsInCurrentRole':YearsInCurrentRole,
                'YearsSinceLastPromotion':YearsSinceLastPromotion,
                'YearsWithCurrManager':YearsWithCurrManager,
                'AgeGroup':AgeGroup,
                'TenureGroup':TenureGroup,
                'MonthlyIncomeBracket':MonthlyIncomeBracket,
                'OverallSatisfactionIndex':OverallSatisfactionIndex
            }
        ]
    )

    try:
        # step 1: predict, 0-stay or 1-leave
        prediction = model.predict(input_data)[0]

        # step 2: get actual probability of leaving
        # [:, 1] extracts the probability of class 1 ("Yes - Attrition")
        churn_probability = model.predict_proba(input_data)[0][1] * 100

        # step 3: present results
        result1, result2 = st.columns([1, 2])

        with result1:
            st.metric(
                label="Attrition Risk Score",
                value=f"{churn_probability:.1f}%",
                delta="High Risk" if churn_probability > 50 else "-Low Risk",
                delta_color="inverse"
            )

        with result2:
            if prediction == 1:
                st.error("🚨 **High Risk of Leaving:** This employee shows high likelihood of quitting.")
            else:
                st.success("✅ **Low Risk / Stable:** This employee is likely to remain at the company.")

        st.info(
            "💡 **Key Retention Drivers Identified by this Model:**\n"
                "* **OverTime** is the single strongest predictor of employee departure.\n"
                "* **Single employees** and those traveling frequently have higher baseline attrition rates.\n"
                "* High **Overall Satisfaction** and longer **Tenure** strongly encourage employees to stay."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")