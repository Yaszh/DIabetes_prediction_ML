import streamlit as st 
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('trained_model.pkl','rb'))

# creating a function for prediction

def diabetes_prediction(input_data):
    input_np =np.asarray(input_data) 

    input_reshape = input_np.reshape(1,-1)
    
   
    prediction = loaded_model.predict(input_reshape)
    print(prediction)

    if(prediction[0]==0):
        return 'You are Non-Diabetic'
    else:
        return 'You are Diabetic'



def main():
    st.title('Diabetes Prediction Web App') 
    
    
  
    st.title('Enter the details:')
    Pregnancies = st.text_input('No of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin')
    BMI= st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Pedigree Function')
    Age = st.text_input('Age')
    
        

    diabetes = ''

        # create a button or predict

    if st.button("Click for results"):
        diabetes = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diabetes)


if __name__=='__main__':
    main()