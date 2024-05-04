import streamlit as st 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()




def get_data(filename):
    tripdata=pd.read_parquet(filename)

    return tripdata

with header:
    st.title('First streamlit data science project ')
    st.text('In this project i look at the transaction of taxis in New York city ')

with dataset:
    st.header('New York Dataset')
    st.text('I found it on KAGGLE ')
    
    tripdata=get_data('data/Yellowtaxitripdata.parquet')
    # st.write(tripdata.head())

    st.subheader('Pick up Location ID distribution on the NYC Dataset ')
    pulocation_dist=pd.DataFrame(tripdata['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with features:
    st.header('Features I Created' )
    st.markdown('* **First Feature:** Describe the reason1')
    st.markdown('* **Second Feature:** Describe the reason2')


with model_training:
    st.header('Time to Train the model ')
    st.text('Here you get to choose the hyperparameters and you see how it changes ')
    
    sel_col,disp_col=st.columns(2)

    max_depth=sel_col.slider('What should be the Max depth of the model?',min_value=10,max_value=100,value=20,step=10)

    n_estimators=sel_col.selectbox('How many Trees should there be ?',options =[100,200,300,],index=0)

    if n_estimators == 'No limit':
        n_estimators = None  # or some large number, like 1000
    else:
        n_estimators = int(n_estimators)

    sel_col.text('Here is a list of the Input Features ')
    sel_col.write(tripdata.columns)

    input_feature=sel_col.text_input('Which feature should be used as an input?','PULocationID')

    regr=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

   

    X=tripdata[[input_feature]]
    y=tripdata['trip_distance']

    y = np.ravel(y)

    regr.fit(X, y)
    prediction= regr.predict(X)

    disp_col.subheader('Mean Absolute Error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean Squared Error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R2 score of the model is:')
    disp_col.write(r2_score(y, prediction))

