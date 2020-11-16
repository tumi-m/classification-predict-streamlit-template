"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
#import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import os
#from pathlib import Pathpip

# Pickle dependencies
import pickle


#Image dependenices
from PIL import Image

st.set_page_config(
    page_title="TEAM_7_JHB classification App",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Vectorizer
news_vectorizer = open("resources/team7_vectorizer.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	    """Tweet Classifier App with Streamlit """

	    #Creates a main title and subheader on your page -
	    # these are static across all pages
	
	    html_temp = """
    	<div style="background:#025246 ;padding:5px">
    	<h2 style="color:white;text-align:center;">TEAM 7 Tweet classification ML App </h2>
    	</div>
    	"""
	    st.markdown(html_temp, unsafe_allow_html = True)
	    st.markdown('<style>body{background-color: #CECEF6;}</style>',unsafe_allow_html=True)

	    #image
	
	    image = Image.open('resources/imgs/Quiver1.jpg')

	    st.image(image,use_column_width=True)
	

	#page_bg_img = '''
	#<style>
	#body {
	#background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
	#background-size: cover;
	#}
	#</style>
	#'''

	#st.markdown(page_bg_img, unsafe_allow_html=True)

	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Background", "Insights", "Prediction", "About Team 7", "Contact Us"]
	selection = st.sidebar.selectbox("Please select an option", options)
	st.markdown(
	"""
	<style>
	.sidebar .sidebar-content{
	background-image: linear-gradient(#025246, #025246);
	font color: white;
	}
	</style>
	""",
	unsafe_allow_html=True,
	)

	#Building out the "Insights" page
	st.write('You selected the :', options)
	if selection == "Insights":
		st.info("General Information")
  

		st.subheader("Raw Twitter data and label")
	if st.radio('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
    


    #Building out the Background
	if selection == "Background":
        	st.warning("We were tasked with developing a robust classification model based on users historical tweet data, that can predict their perception of sustainability in companys' products and services")

	if st.checkbox("Introduction"):
			st.subheader("Introduction to Classification Predict")
			st.error("There is an excess of information about climate change on social media platforms like Twitter. As such it is difficult for companies to discern the beliefs of people relating to climate change. Having a robust machine learning model that is able to classify a person's belief in climate change from historical tweet data, as part of companies research toolset will enable them to better gauge their perception of customers relating to sustainability their products and services.")


	#Building out the About Page
	if selection == "About Team 7":
		st.subheader("TEAM 7 is a group of four members from EDSA comprising of Thiyasize Kubeka, Warren Mnisi, Samuel Aina, and Tumelo Malebo")
		st.subheader("Visit our Contact Page and lets get in touch!")


	#Building out the Contact Page
	if selection == "Contact Us":
		st.info("Lets get in touch for all your ML needs")
		firstname = st.text_input("Enter your Name", "Type Here Please...")
		lastname = st.text_input("Enter your last Name", "Type Here Please..")
		contactDetails = st.text_input("Enter your contact details here", "Type Here Please...")
		message = st.text_area("Tell us about your compaby's Data Science needs", "Type here Please..")
  
		if st.button("Submit"):
			result = message.title()
		st.success(result)


	if selection == "Prediction":
	#st.info("Prediction with ML Models")
	# Creating a text box for user input
		options = st.selectbox(
			'Which model would you like to use?',
			('KNN Model', 'Linear SVC Model', 'Complement Naive Bayes Model'))

		st.write('You selected the :', options)
		


		if options == 'KNN Model':
				st.warning("KNN or K-nearest Neighbors Classifiers algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other")
				# Creating a text box for user input
				user_text = st.text_area("Enter Text","Type Here")

				if st.button("Classify"):
					# Transforming user input with vectorizer
					#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
					KNN= joblib.load(open(os.path.join("resources/team7_KNN.pkl"),"rb"))
					prediction = KNN.predict([user_text])
					st.success("Text Categorized as : {}".format(prediction))

					if prediction[0] == 2:
						st.info('This tweet links to factual news about climate change')
					if prediction[0] == 1:
						st.success('Tweet support the believe of man-made climate change')
					if prediction[0] == 0:
						st.warning('Tweet neither supports nor refutes the believe of man-made climate change')
					if prediction[0] == -1:
						st.error('Tweet does not believe in man-made climate change')
      
		
		if options == 'Linear SVC Model':
				st.warning("A linear SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.")
				# Creating a text box for user input
				user_text = st.text_area("Enter Text","Type Here")
				

				if st.button("Classify"):
					# Transforming user input with vectorizer
					#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
					Linear_SVC = joblib.load(open(os.path.join("resources/team7_linear_svc.pkl"),"rb"))
					prediction = Linear_SVC.predict([user_text])
					st.success("Text Categorized as : {}".format(prediction))

					if prediction[0] == 2:
						st.info('Tweet links factual news about climate change')
					if prediction[0] == 1:
						st.success('Tweet support the believe of man-made climate change')
					if prediction[0] == 0:
						st.warning('Tweet neither supports nor refutes the believe of man-made climate change')
					if prediction[0] == -1:
						st.error('Tweet does not believe in man-made climate change')
      

        
		if options == 'Complement Naive Bayes Model':
				st.warning("Complement Naive Bayes is particularly suited to work with imbalanced datasets. In complement Naive Bayes, instead of calculating the probability of an item belonging to a certain class, we calculate the probability of the item belonging to all the classes.")
				# Creating a text box for user input
				user_text = st.text_area("Enter Text","Type Here")
				
				if st.button("Classify"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
					LR = joblib.load(open(os.path.join("resources/team7_complement_naive_bayes_model.pkl"),"rb"))
					prediction = LR.predict([user_text])
					st.success("Text Categorized as : {}".format(prediction))

					if prediction[0] == 2:
							st.info('This tweet links to factual news about climate change')
					if prediction[0] == 1:
							st.success('This tweet supports the belief of man-made climate climate')
					if prediction[0] == 0:
							st.warning('This tweet neither refutes or supports man-made climate change')
					if prediction[0] == -1:
							st.error(' This tweet does not believe in man-made climate change')
      
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
 
 
 
        		#options = st.selectbox(
					#'Which model would you like to use?',
					#('word cloud for pro and news', 'word cloud for anti and neutral', 'Tweet distribution by sentiment', 'Emoji analysis', 'Overall hashtag analysis', 'Retweet distribution by sentiment'))

		   		#st.write('You selected the :', options)
		   
