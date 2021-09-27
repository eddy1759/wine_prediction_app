import numpy as np 
import pandas as pd 
import sklearn
import joblib
import streamlit as st
from xgboost import XGBClassifier
from PIL import Image


model = joblib.load(open('wine_model.joblib','rb'))
print('Loading model........')
print('Model succesfully loaded')


#st.beta_set_page_config(
 #   page_title="Wine Prediction App",
  #  page_icon="🤖",
   # layout="centered",
    #initial_sidebar_state="expanded"
    #)

st.write("""
# Wine Quality Prediction ML Web-App 
This app predicts the ** Quality of Wine **  using **wine features** 
""")

#image = Image.open('wine.jpeg')
#st.image(image, caption='wine', use_colum_width = 'always')    

image = Image.open('wine.jpeg')
st.image(image,use_column_width=True)


def hello():
    return "your're all welcome"


def prediction(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):

    input = np.array([[fixed_acidity, volatile_acidity, citric_acid, 
      residual_sugar, chlorides, free_sulfur_dioxide, 
      total_sulfur_dioxide, density, pH, sulphates, alcohol]]).astype(np.float64)

    pred = int(model.predict(input))
    if pred == 0:
        pred = 'Low'
    elif pred == 1:
        pred  = 'Medium'
    else:
        pred = 'High'    
    return pred


def main():
    st.title('Wine Quality Prediction')
    html_temp = """ 
    <div style ="background-color:red;padding:13px"> 
    <h2 style ="color:white;text-align:center;">Wine Quality Prediction ML App</h2> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    fixed_acidity = st.text_input('Fixed Acidity', 'Type Here')
    volatile_acidity = st.text_input('Volatile Acidity', 'Type Here')
    citric_acid = st.text_input('Citric Acid', 'Type Here')
    residual_sugar = st.text_input('Residual Sugar', 'Type Here')
    chlorides = st.text_input('Chlorides', 'Type Here')
    free_sulfur_dioxide = st.text_input('Free Sulphur Dioxide', 'Type Here')
    total_sulfur_dioxide = st.text_input('Total Sulphur Dioxide', 'Type Here')
    density = st.text_input('Density', 'Type Here')
    pH = st.text_input('pH', 'Type Here')
    sulphates = st.text_input('Sulphates', 'Type Here')
    alcohol = st.text_input('Alcohol', 'Type Here')
    
    safe_html = """
    <div style="background-color:#80ff80; padding:13px >
      <h2 style="color:white;text-align:center;"> The Wine is high in quality</h2>
      </div>
    """
    warn_html ="""  
      <div style="background-color:#F4D03F; padding:13px >
      <h2 style="color:red;text-align:center;"> The Wine has an average qualitye</h2>
      </div>
     """

    danger_html="""  
      <div style="background-color:#F08080; padding:13px >
       <h2 style="color:black ;text-align:center;"> The Wine is low in quality</h2>
       </div>
    """   

    if st.button('Predict the wine quality'):

        output = prediction(fixed_acidity, volatile_acidity, 
        citric_acid, residual_sugar, chlorides, 
        free_sulfur_dioxide, total_sulfur_dioxide, density, 
        pH, sulphates, alcohol)

        st.success(f'The wine quality is {output}')

        if output == 0:
            st.markdown(safe_html,unsafe_allow_html=True)
        elif output == 1:
            st.markdown(warn_html,unsafe_allow_html=True)
        elif output == 2:
            st.markdown(danger_html,unsafe_allow_html=True)



if __name__=='__main__':
    main()