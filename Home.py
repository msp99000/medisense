import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load(open('./models/model.pkl', 'rb'))

# Load the label encoders and other preprocessing files if necessary

# Define the feature options
age_options = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
year_options = ['2016', '2017', '2018', '2019', '2020']
grade_options = ['Grade I', 'Grade II', 'Grade III']
yes_no_options = ['No', 'Yes']
rucc_options = ['Urban', 'Non-Urban']
psite_options = ['C504','C508','C502','C505','C503','C509','C501']
hist_options = ['8500', '8520', '8522', '8480']
lvi_options = ['Present', 'Not Present', 'Unknown']
lateral_options = ['Left: origin of primary', 'Right: origin of primary', 'Paired site, but no information concerning laterality']
pri_options = [
    'Private Insurance: Managed care, HMO, or PPO',
    'Medicare - Administered through a Managed Care plan',
    'Medicare with private supplement',
    'Insurance, NOS',
    'Medicare with supplement, NOS',
    'Medicaid - Administered through a Managed Care plan',
    'Medicare/Medicare, NOS',
    'Medicaid',
    'Private Insurance: Fee-for-Service',
    'Medicare with Medicaid eligibility',
    'Insurance status unknown',
    'TRICARE',
    'Not insured, self-pay',
    'Not insured',
    'Veterans Affairs',
    'Military'   
]


# Define the web app interface
st.title('MediSense')
st.header('Breast Cancer Prediction')
st.subheader('Enter the required information to predict breast cancer.')

# Split the inputs into two columns
col1, col2 = st.columns(2)

# Input fields - Column 1
with col1:
    age = st.selectbox('Age', age_options)
    year = st.selectbox('Year', year_options)
    grade = st.number_input('Grade', min_value=1, max_value=3, step=1)
    psite = st.selectbox('PSite', psite_options)
    hist_type = st.selectbox('HistTypeICDO3', hist_options)
    lateral = st.selectbox('Lateral', lateral_options)
    lvi = st.selectbox('LVI', lvi_options)
    pri_payer = st.selectbox('PriPayerDx', pri_options)
    time_to_rad = st.number_input('Time to Radiation', step=1)
    time_to_chemo = st.number_input('Time to Chemotherapy', step=1)
    fips = st.number_input('FIPS', step=1)


# Input fields - Column 2
with col2:
    vital_status = st.selectbox('Vital Status', yes_no_options)
    rucc = st.selectbox('RUCC', rucc_options)
    physician = st.number_input('Physician', step=1)
    hospital_bed = st.number_input('HospitalBed', step=1)
    rad_flag = st.selectbox('Rad Flag', yes_no_options)
    surg_flag = st.selectbox('SurgFlag', yes_no_options)
    chemo_flag = st.selectbox('Chemo Flag', yes_no_options)
    horm_flag = st.selectbox('Horm Flag', yes_no_options)
    immuno_flag = st.selectbox('Immuno Flag', yes_no_options)
    other_flag = st.selectbox('Other Flag', yes_no_options)
    time_to_survival = st.number_input('Time to Survival', step=1)




# Prepare the input data
input_data = pd.DataFrame({
    'Age': [age],
    'YEAR': [year],
    'Grade': [grade],
    'PSite': [psite],
    'HistTypeICDO3': [hist_type],
    'Lateral': [lateral],
    'LVI': [lvi],
    'PriPayerDx': [pri_payer],
    'VITALSTATUS': [vital_status],
    'RUCC': [rucc],
    'Physician': [physician],
    'HospitalBed': [hospital_bed],
    'Rad_Flag': [rad_flag],
    'Surg_Flag': [surg_flag],
    'Chemo_flag': [chemo_flag],
    'Horm_flag': [horm_flag],
    'Immuno_flag': [immuno_flag],
    'Other_flag': [other_flag],
    'TimetoRad': [time_to_rad],
    'TimetoChemo': [time_to_chemo],
    'FIPS': [fips],
    'timetosurvival': [time_to_survival]
})

input_data = pd.get_dummies(input_data)

# Display the prediction
st.subheader('Prediction')

if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(input_data)

    st.write('The predicted result is:', prediction[0])

