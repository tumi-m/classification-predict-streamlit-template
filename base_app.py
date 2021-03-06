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

	html_temp = """
    <div style="background:#025246 ;padding:5px">
    <h2 style="color:white;text-align:center;">TEAM 7 Tweet classification ML App </h2>
    </div>
    """
	st.markdown(html_temp, unsafe_allow_html = True)
	st.markdown('<style>body{background-color: #D8D8D8;}</style>',unsafe_allow_html=True)

	#image
	
	image = Image.open('resources/imgs/Quiver1.jpg')

	st.image(image,use_column_width=True)
	
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



		#Building out the About Page
	if selection == "About Team 7":
		st.subheader("TEAM 7 is a group of four members from EDSA comprising of Thiyasize Kubeka, Warren Mnisi, Samuel Aina, and Tumelo Malebo")
		st.subheader("Visit our Contact Page and lets get in touch!")
  
	#Building out the "Insights" page   
	if selection == "Insights":
      	#st.warning("Please view the visualations and see how insightful the raw dataset became!")
       #Word cloud for anti and neutral tweets
       
		if st.checkbox("Word cloud for pro and news"):
			st.subheader("Word cloud analysis for pro and factual sentiments")
			st.image('resources/imgs/Word_Cloud_First_line.PNG')
   
		#Word cloud for anti and neutral tweets
		if st.checkbox("Word cloud for anti and neutral"):
			st.subheader("Word cloud analysis for anti and neutral sentiments")
			st.image('resources/imgs/Word_Cloud_Second_line.PNG')
   
		#Tweet distribution by sentiment
		if st.checkbox("Tweet distribution by sentiment"):
			st.subheader("count of words per each sentiment with clean data")
			st.image('resources/imgs/Distribution_of_Tweets_per_Sentiment_Group.PNG', channels="BGR")

		#Emoji analyis
		if st.checkbox("Emoji analysis"):
			st.subheader("A detailed analysis of the emoji's within the dataset")
			st.image('resources/imgs/Emoji_Analysis.PNG', channels="BGR")

		#Overall hashtag analysis
		if st.checkbox("Overall hashtag analysis"):
			st.subheader("This is an overall view of the hashtags encompasses the Pro, News, Neutral and Anti classes")
			st.image('resources/imgs/Hashtag_Analysis_Overall.PNG', channels="BGR")
			
			#Retweet distribution by hashtag
		if st.checkbox("Retweet distribution by sentiment"):
			st.subheader('frequent retweets are distributed to showcase which ones are the most common')
			st.image('resources/imgs/Tweet - Retweet Distributions by Sentiment Class.PNG', channels="BGR")
      
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.subheader('Raw Twitter data and label')
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
    
    
  
	if selection == "Background":
		
		st.success("We built a robust machine learning classification model to help tweets to be better classified wether they believe climate change is a man-made phenomena. In turn, this will help the market research divisions in offering more sustainable products and services.")

		st.write('Introduction')
		st.success("Climate change always been acknowledged by the scientific community as one of the critical issues facing mankind. Many companies are built around the idea of preserving an environmental friendly culture, be it on the products and services they offer or the actual raw material used to extend the structure of their company. One thing that has not been of certain yet is what perception other people have on environmental awareness.It is of importance to understand public perception of climate change through the use of twitter data. This will allow companies access to a broad base of consumer sentiment thereby increasing their insights for future marketing strategies.")
		
		st.write("Problem Statement")
		st.success("There is an excess of information about climate change on social media platforms like Twitter. As such it is difficult for companies to discern the beliefs of people relating to climate change. Having a robust machine learning model that is able to classify a person's belief in climate change from historical tweet data, as part of companies research toolset will enable them to better gauge their perception of customers relating to sustainability their products and services.")

		st.write("Conclusion")
		st.success("In conclusion we found that most tweets were from users who believed that climate change is a man-made phenomena. As such, for company's to stay relevant with most users is required to adapt their business models to more sustainable products and services as well being mindful of the externalities they bring along.")


	#Building out the Contact Page
	if selection == "Contact Us":
		st.info("Lets get in touch for all your ML needs")
		firstname = st.text_input("Enter your Name", "Type Here Please...")
		lastname = st.text_input("Enter your last Name", "Type Here Please..")
		contactdetails = st.text_input("Enter your contact details here", "Type Here Please...")
		message = st.text_area("Tell us about your compaby's Data Science needs", "Type here Please..")
  
		if st.button("Submit"):
			result = message.title()
			st.success(result)


	#Building out the Prediction Page
	if selection == "Prediction":
	# Creating a text box for user input
		options = st.selectbox(
			'Which model would you like to use?',
			('KNN Model', 'Linear SVC Model', 'Complement Naive Bayes Model'))

		st.write('You selected the :', options)
		#Selecting the KNN model
		if options == 'KNN Model':
			st.success("KNN or K-nearest Neighbors Classifiers algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other")
			# Creating a text box for user input
		user_text = st.text_area("Enter Text","Type Here", key=1)

		if st.button("Classify", key=1):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
			KNN= joblib.load(open(os.path.join("resources/team7_KNN.pkl"),"rb"))
			prediction = KNN.predict([user_text])
			st.success("Text Categorized as : {}".format(prediction))
			#Climate change Class result
			if prediction[0] == 2:
				st.info('This tweet links to factual news about climate change')
			if prediction[0] == 1:
				st.success('Tweet support the believe of man-made climate change')
			if prediction[0] == 0:
				st.warning('Tweet neither supports nor refutes the believe of man-made climate change')
			if prediction[0] == -1:
				st.error('Tweet does not believe in man-made climate change')
      
			#Selecting the Linear SVC Model
			if options == 'Linear SVC Model':
				st.success("A linear SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.")
				# Creating a text box for user input
			user_text = st.text_area("Enter Text","Type Here", key=2)
				

			if st.button("Classify", key=2):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
				Linear_SVC = joblib.load(open(os.path.join("resources/team7_linear_svc.pkl"),"rb"))
				prediction = Linear_SVC.predict([user_text])
				st.success("Text Categorized as : {}".format(prediction))
				#Climate change Class result
				if prediction[0] == 2:
					st.info('Tweet links factual news about climate change')
				if prediction[0] == 1:
					st.success('Tweet support the believe of man-made climate change')
				if prediction[0] == 0:
					st.warning('Tweet neither supports nor refutes the believe of man-made climate change')
				if prediction[0] == -1:
					st.error('Tweet does not believe in man-made climate change')
      

			#Selecting the Complement Naive Bayes Model
			if options == 'Complement Naive Bayes Model':
				st.success("Complement Naive Bayes is particularly suited to work with imbalanced datasets. In complement Naive Bayes, instead of calculating the probability of an item belonging to a certain class, we calculate the probability of the item belonging to all the classes.")
					# Creating a text box for user input
			user_text = st.text_area("Enter Text","Type Here", key=3)
				
			if st.button("Classify", key=3):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.fit_transform([tweet_text]).toarray()
				LR = joblib.load(open(os.path.join("resources/team7_complement_naive_bayes_model.pkl"),"rb"))
				prediction = LR.predict([user_text])
				st.success("Text Categorized as : {}".format(prediction))
				#Climate change Class result
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