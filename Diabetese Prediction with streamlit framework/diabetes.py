import streamlit as st
import pandas as op
import pickle 
import warnings
warnings.filterwarnings("ignore")


#Load Model
with open('model.pkl', 'rb') as fp:
    classifier=pickle.load(fp)

with open('std_s.pkl', 'rb') as fp1:
    s_s=pickle.load(fp1)


def main():

    st.title("Diabetes Predection")
    left,right=st.columns((2,2))
    Pregnancies=left.number_input("Enter your pregnany count as Whole number",step=1,value=0)
    Glucose=right.number_input("Enter your glucose count as Whole number",step=1,value=0)
    BloodPressure=left.number_input("Enter your bloodpressure count as Whole number",step=1,value=0)
    SkinThickness=right.number_input("Enter your skinthickness count as Whole number",step=1,value=0)
    Insulin=left.number_input("Enter your Insulin count as Whole number",step=1,value=0)
    BMI=right.number_input("Enter your BodyMassIndex count as decimals")
    DiabetesPedigreeFunction=left.number_input("Enter your DiabetesPedigreeFunction count as decimals")
    Age=right.number_input("Enter your Age as Whole number",step=1,value=0)

    pred=st.button("Am I a Daibetic")

    scaled_data=s_s.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    if pred:
        res=classifier.predict(scaled_data)
        if(res[0]==0):
            st.success("You are Not Daibetic")
            st.balloons()
        else:
            st.success("You are Daibetic")
            st.warning("Please take care of your health",icon="🚨")

if __name__=="__main__":
    main()