#https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sbrn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

# Read the dataset
data_var = pd.read_csv('datafile/exam_new.csv')

# Save columns
X = data_var[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']]
y = data_var['math_score']

# Categorical columns
categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

# Preprocessor for categorial cols
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)

# Description Method
# Model - Linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Data split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model.fit(X_train, y_train)

# Test Model Evaluation
test_score = model.score(X_test, y_test)
print(f"R-squared score: {test_score:.2f}")

# Trained Model Evaluation
train_score = model.score(X_train, y_train)
print(f"R-squared score: {train_score:.2f}")

# print(X_test)
# print(y_test)
# exit()

# Model Function
def model_analyze(gender_input_value, race_input_value, parent_input_value, lunch_input_value, prep_input_value, reading_score_value, writing_score_value):
    new_sample = pd.DataFrame({
        'gender': [gender_input_value],
        'race_ethnicity': [race_input_value],
        'parental_level_of_education': [parent_input_value],
        'lunch': [lunch_input_value],
        'test_preparation_course': [prep_input_value],
        'reading_score': reading_score_value,
        'writing_score': writing_score_value,
    })

    predicted_score = model.predict(new_sample)
    return predicted_score
    
# Set Page layout and header
st.set_page_config(layout='wide')
st.header('Student Performance Analyzer')


# Init dropdown values
gender_list = ['male', 'female']
race_list = ['group A', 'group B', 'group C', 'group D', 'group E']
parent_edu_list = ['bachelor\'s degree', 'some college', 'master\'s degree', 'associate\'s degree', 'high school']
lunch_list = ['standard', 'free/reduced']
prep_list = ['completed', 'none']

# Create select dropdowns
gender_input_value = st.selectbox('Choose Student\'s Gender', gender_list)
race_input_value = st.selectbox('Choose Student\'s Race / Ethnicity Group', race_list)
parent_input_value = st.selectbox('Choose Student\'s parent\'s education level', parent_edu_list)
lunch_input_value = st.selectbox('Choose Student\'s lunch preference before taking the test', lunch_list)
prep_input_value = st.selectbox('Choose Student plants to take the preparation course', prep_list)
reading_score_value = st.number_input("Reading Score", value=50, min_value=0, max_value=100, placeholder="Type a number...")
writing_score_value = st.number_input("Writing Score", value=50, min_value=0, max_value=100, placeholder="Type a number...")

# Button Action
if st.button('Analyze'):

    # Run Model Ffunction
    predicted_score_return = model_analyze(gender_input_value, race_input_value, parent_input_value, lunch_input_value, prep_input_value, reading_score_value, writing_score_value)

    # Show Result Log
    print(f"New Predicted math score: {predicted_score_return[0]:.2f}")
    # Return Result
    output_value = f"New Predicted math score: {predicted_score_return[0]:.2f}"

    st.header("Analyzed Result")
    # Output Result
    cols = st.columns(1)
    for col in cols:
        col.write(output_value)
        

st.divider()

st.header("Graphs")
# graphs from below

# Group by 'parental_level_of_education' and calculate the average math score
average_scores = data_var.groupby('parental_level_of_education')['math_score'].mean()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(average_scores.index, average_scores.values, color='maroon')
plt.xlabel("Parent Education Level")
plt.ylabel("Average Math Score")
plt.title("Average Math Scores by Parent Education Level")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
st.pyplot(plt.gcf(), 1, 0)
# st.pyplot(plt.gcf())
plt.close()


# Group by 'race' and calculate the average math score
average_race = data_var.groupby('race_ethnicity')['math_score'].mean()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(average_race.index, average_race.values, color='maroon')
plt.xlabel("race_ethnicity")
plt.ylabel("Average Math Score")
plt.title("Average Math Scores by race_ethnicity")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
st.pyplot(plt.gcf(), 1, 0)
plt.close()


# scatter graph 1
scatterplot_val = sbrn.scatterplot(x='reading_score', 
                y='math_score', data=data_var)

st.pyplot(scatterplot_val.get_figure(), 1, 0)
plt.close()

# scatter graph 2
scatterplot_val = sbrn.scatterplot(x='writing_score', 
                y='math_score', data=data_var)

st.pyplot(scatterplot_val.get_figure(), 1, 0)
plt.close()



# End