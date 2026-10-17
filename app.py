# Titanic Passenger Survival Prediction System - Streamlit version 
# user input -> feature vector -> label encoding -> SVC prediction.
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")


PCLASS = ["First", "Second", "Third"]
GENDER = ["Male", "Female"]
SIBLING = ["Zero", "One", "Two", "Three"]
EMBARKED = ["Southampton", "Cherbourg", "Queenstown"]

# Recreate the label encoders exactly like the notebook (fit on the same lists)
pclass_label_encoder = LabelEncoder().fit(np.ravel(PCLASS))
gender_label_encoder = LabelEncoder().fit(np.ravel(GENDER))
sibling_label_encoder = LabelEncoder().fit(np.ravel(SIBLING))
embarked_label_encoder = LabelEncoder().fit(np.ravel(EMBARKED))


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svc_trained_model.pkl")
    with open(path, "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("🚢 Titanic Passenger Survival Prediction")


#  take input from the user
pclass_input = st.selectbox("Passenger class (PClass)", PCLASS)
gender_input = st.selectbox("Gender", GENDER)
sibling_input = st.selectbox("Siblings / spouses aboard (Sibling)", SIBLING)
embarked_input = st.selectbox("Port of embarkation (Embarked)", EMBARKED)

if st.button("Predict", type="primary"):
    #  convert user input into a feature vector (same as the sample data)
    user_input = pd.DataFrame({
        "PClass": [pclass_input],
        "Gender": [gender_input],
        "Sibling": [sibling_input],
        "Embarked": [embarked_input],
    })


    unseen_data_features = user_input.copy()
    unseen_data_features["PClass"] = pclass_label_encoder.transform(user_input["PClass"])
    unseen_data_features["Gender"] = gender_label_encoder.transform(user_input["Gender"])
    unseen_data_features["Sibling"] = sibling_label_encoder.transform(user_input["Sibling"])
    unseen_data_features["Embarked"] = embarked_label_encoder.transform(user_input["Embarked"])


    predicted_survival = model.predict(unseen_data_features)[0]

    st.write("Input:", user_input.iloc[0].to_dict())
    if predicted_survival == 1:
        st.success("SURVIVED")
    else:
        st.error("NOT SURVIVED")