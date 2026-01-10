import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('models/model.h5')

with open('models/onehot_encoder_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('models/label_encoder_gender.pkl', 'rb') as file:
    label_gender = pickle.load(file)

with open('mdoels/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



## Stramlit app

st.title('Customer Churn Prediction')

# Input
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
estimated_salary = st.number_input('Estimated salary')
creditscore = st.number_input('Credit Score')
tenure = st.slider('Tenure')
num_of_products = st.slider('Number of Products', 0, 10)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Active Memeber?', [0, 1])

# input dictionary:
input_data = pd.DataFrame({
    'CreditScore' : [creditscore],
    'Gender' : [label_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# geo data

geo_data_encoded = ohe_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_data_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# merging df

df = pd.concat([input_data.reset_index(drop = True), geo_df], axis = 1)

# Scaled df

scaled_df = scaler.transform(df)

#prediction
prediction = model.predict(scaled_df)
prediction_prob = prediction[0][0]

st.write(prediction_prob)

if prediction_prob > 0.5:
    st.write('The customer is likly to leave the Bank')
else:
    st.write('The customer will not leave the Bank')