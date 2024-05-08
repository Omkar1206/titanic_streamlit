#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_pred.pkl')

# Main function to run the Streamlit app
def main():
    st.title('Titanic Survival Prediction')
    
    # Sidebar for user inputs
    st.sidebar.title('User Inputs')
    pclass = st.sidebar.slider('Select Passenger Class (1st, 2nd, 3rd)', 1, 3, 3)
    age = st.sidebar.slider('Select Age', 0, 100, 30)
    siblings_spouses = st.sidebar.slider('Select Number of Siblings/Spouses Aboard', 0, 8, 0)
    parents_children = st.sidebar.slider('Select Number of Parents/Children Aboard', 0, 6, 0)
    fare = st.sidebar.slider('Select Fare', 0, 600, 30)
    sex = st.sidebar.radio('Select Gender', ['Male', 'Female'])
    embarked = st.sidebar.radio('Select Embarked Location', ['Queenstown', 'Southampton', 'Cherbourg'])
    
    sex = 1 if sex == 'Male' else 0
    embarked_Q = 1 if embarked == 'Queenstown' else 0
    embarked_S = 1 if embarked == 'Southampton' else 0
    
    # Predict survival
    prediction = model.predict([[pclass, age, siblings_spouses, parents_children, fare, sex, embarked_Q, embarked_S]])
    
    # Display the prediction
    st.subheader('Prediction')
    if prediction[0] == 0:
        st.write('The model predicts that the passenger did not survive.')
    else:
        st.write('The model predicts that the passenger survived.')

if __name__ == '__main__':
    main()


# In[ ]:




