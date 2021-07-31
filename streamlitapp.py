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

def main():

    df = load_data(r"C:\Users\DELL\sales\data\train_store.csv")
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title('Rossman Sales Prediction')
        st.text('Select a page in the sidebar')
        st.dataframe(df)
    elif page == 'Exploration':
        st.title('Exploration of Rossman Sales Dataset')
        if st.checkbox('Show column descriptions'):
            st.dataframe(df.describe())
        
        st.markdown('### Analysing column relations')
        st.text('Correlations:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot()
        st.text('Effect of the different classes')
        sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='Sales')
        st.pyplot()
    else:
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.markdown('### Make prediction')
        st.dataframe(df)
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Predicted')
        st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


@st.cache
def train_model(df):
    features = ["Store",'DayOfWeek','Promo',
       'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'DayOfMonth',
       'StoreType', 'Assortment',
       'CompetitionDistance', 'Promo2',
       'CompetitionOpen', 'PromoOpen']
    X = df[features]
    y= df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)

@st.cache
def load_data(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    main()

