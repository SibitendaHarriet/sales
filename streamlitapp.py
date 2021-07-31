import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()
st.title('Analysis of sales data')
st.write("Here's our first attempt at using data to create a table:")

with dataset:
	st.header('Pharmacutical sales dataset')
	st.text('I gathered data from users ability to acess data and network ')

	text_data = pd.read_csv(r"C:\Users\DELL\sales\data\train_store.csv")
	st.write(text_data.head())

	st.subheader('Exploration of Rossman Sales Dataset')
	User_interaction = pd.DataFrame(text_data['Sales'].value_counts())
    #sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='Sales')
	#st.pyplot()
    #st.bar_chart(User_interaction)
    
	st.subheader('Analysing column relations')
	User_interactionsp = pd.DataFrame(text_data['CompetitionDistance'].value_counts()).head(50)
	st.bar_chart(User_interactionsp)

with features:
	st.header('The features I created')
	st.markdown('* ** The first features I created were about data access on different applications')
	st.markdown('* ** The second features I created were about categorising data')

#with model_training:
	#st.header('Time to train the model!')
	#st.text('Here is my model that was used to train our dataset using Random Forest Classifier')
	#st.write('Accuracy: ' + str(accuracy))
    #st.markdown('### Make prediction')
    #st.dataframe(df)
    #row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
    #st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])
  