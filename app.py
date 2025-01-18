import streamlit as st #used to create interactive web application.
import pandas as pd #Handles data manipulation and preprocessing.
import numpy as np #Provides numerical operations and array manipulation.
import base64 #Encodes and decodes binary data, used for setting a background image.
from sklearn.ensemble import RandomForestRegressor #Predicts the time to tumor recurrence.
from sklearn.preprocessing import LabelEncoder #Converts categorical features into numerical format for model input.

# Load and preprocess the data
data = pd.read_csv('BrainTumor.csv') # The dataset is loaded from a CSV file called BrainTumor.csv.

# Drop rows with missing 'Time to Recurrence (months)' as this is our target variable
data_clean = data.dropna(subset=['Time to Recurrence (months)'])
#Removes rows where the target variable (Time to Recurrence (months)) is missing to ensure the model trains on complete data.


# Encode categorical features
label_encoders = {} #A dictionary to store LabelEncoder objects for each categorical feature.
categorical_features = ['Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location', 'Treatment', 'Treatment Outcome'] #Lists the categorical columns in the dataset.

#Converts categorical values (like "Male", "Female") into numerical values (e.g., 0, 1).
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data_clean[feature] = label_encoders[feature].fit_transform(data_clean[feature])

# Define features and target variable
features = ['Age', 'Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location', 'Treatment', 'Treatment Outcome'] #List of independent variables used for prediction.
target = 'Time to Recurrence (months)' #: Dependent variable, the time to recurrence in months.

X = data_clean[features] #Predictor variables (independent).
y = data_clean[target] #Target variable (dependent).

# Initialize and train the model
model = RandomForestRegressor(random_state=42) #A robust regression algorithm that uses an ensemble of decision trees. #Ensures reproducibility by fixing the randomness.
model.fit(X, y) #Trains the model using X (features) and y (target).

# Streamlit app
st.title('TUMOR RECURRENCE PREDICTION') #Displays the app's title on the Streamlit interface.
def set_background(png_file):  #Custom function to add a background image to the app. The image file
    with open(png_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('image.jpg') #is read, encoded with base64, and applied using HTML and CSS styling.


# Input fields
#User Input: Streamlit widgets collect user input for:
#Age (number input).
#Gender, Tumor Type, Tumor Grade, Tumor Location, Treatment, and Treatment Outcome (dropdowns populated with the corresponding encoder classes).
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
tumor_type = st.selectbox('Tumor Type', label_encoders['Tumor Type'].classes_)
tumor_grade = st.selectbox('Tumor Grade', label_encoders['Tumor Grade'].classes_)
tumor_location = st.selectbox('Tumor Location', label_encoders['Tumor Location'].classes_)
treatment = st.selectbox('Treatment', label_encoders['Treatment'].classes_)
treatment_outcome = st.selectbox('Treatment Outcome', label_encoders['Treatment Outcome'].classes_)

# Convert inputs to numerical values
# Converts user-provided categorical inputs into numerical format using transform method of LabelEncoder.
gender_encoded = label_encoders['Gender'].transform([gender])[0]
tumor_type_encoded = label_encoders['Tumor Type'].transform([tumor_type])[0]
tumor_grade_encoded = label_encoders['Tumor Grade'].transform([tumor_grade])[0]
tumor_location_encoded = label_encoders['Tumor Location'].transform([tumor_location])[0]
treatment_encoded = label_encoders['Treatment'].transform([treatment])[0]
treatment_outcome_encoded = label_encoders['Treatment Outcome'].transform([treatment_outcome])[0]

# Predict recurrence time
input_data = np.array([[age, gender_encoded, tumor_type_encoded, tumor_grade_encoded, tumor_location_encoded, treatment_encoded, treatment_outcome_encoded]]) # Combines all inputs into a single array for model prediction.
recurrence_time = model.predict(input_data) #Predicts the time to tumor recurrence.

st.subheader(f'This patients anticipated time to recur (in months) is: {recurrence_time[0]:.2f}') # Displays the prediction on the Streamlit interface.
