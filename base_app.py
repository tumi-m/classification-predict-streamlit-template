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
news_vectorizer = open("resources/team7_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	#Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("TEAM 7 Tweet Classifer")
	#st.subheader("Climate change tweet classification")
	#st.markdown(
    #"""
	#<style>
	#.title .subheader {
    #color: white;
	#text-size: 50%;
	#}
	#</style>
	#""",
    #unsafe_allow_html=True,
	#)
	html_temp = """
    	<div style="background:#025246 ;padding:5px">
    	<h2 style="color:white;text-align:center;">TEAM 7 Tweet classification ML App </h2>
    	</div>
    	"""
	st.markdown(html_temp, unsafe_allow_html = True)


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
	options = ["Prediction", "Information", "Insights", "About Team 7", "Contact Us"]
	selection = st.sidebar.selectbox("Please selection an option", options)
	st.markdown(
	"""
	<style>
	.sidebar .sidebar-content {
	background-image: linear-gradient(#025246, #025246);
	font color: white;
	}
	</style>
	""",
	unsafe_allow_html=True,
	)

	#Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		#dict_check = st.checkbox("Data Dictionary")
		#dict_markdown = read_markdown_file("resources/info.md")

		#if dict_check:
    			#st.markdown(dict_markdown, unsafe_allow_html=True)
				#st.markdown("resources/info.md", unsafe_allow_html=True)
    
		#def read_markdown_file(markdown_file):
    			#return Path(markdown_file).read_text()

		#intro_markdown = read_markdown_file("introduction.md")
		#st.markdown(intro_markdown, unsafe_allow_html=True)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page
    
    
    #Building out the Insights page
    #if selection == "Insights":
    	#st.subheader("Tweet distribution")
        
    #style.use('seaborn-pastel')
	#fig, axes = plt.subplots(ncols=2, 
                         	#nrows=1, 
                         	#figsize=(16, 8), 
                         	#dpi=100)
		#sns.countplot(train_eda['sentiment'], ax=axes[0])

		#labels=['Pro', 'News', 'Neutral', 'Anti'] 
		#axes[1].pie(train_eda['sentiment'].value_counts(),
            #autopct='%1.0f%%',
            #labels=labels,
            #startangle=90,
            #explode = (0.1, 0.1, 0.1, 0.1),
            #shadow=True
           #)
		#fig.suptitle('Distribution of Tweets per Sentiment Group', fontsize=18)
		#plt.show()

	#Building out the About Page
	if selection == "About Team 7":
	#st.info("TEAM 7")
		st.subheader("TEAM 7 is a group of four members from EDSA comprising of Thiyasize Khubeka, Warren Mnisi, Samuel Aina, and Tumelo Malebo")
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



	#with st.beta_container()
	if selection == "Prediction":
	#st.info("Prediction with ML Models")
	# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		options = st.selectbox(
			'Which model would you like to use?',
			('Kernel SVC Model', 'Linear SVC Model', 'Logisitic Regression Model', 'Complement Naive Bayes Model'))

		st.write('You selected the:', options)
  
		if st.button("Classify"):
		# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Try loading in multiple models to give the user a choice
			#options = ["predictor1, predictor2, predictor3, predictor4"]
		

			Kernel_SVC = joblib.load(open(os.path.join("resources/team7_kernel_svc_model.pkl"),"rb"))
			prediction_1 = Kernel_SVC.predict(vect_text)
			Linear_SVC = joblib.load(open(os.path.join("resources/team7_linear_svc_model.pkl"),"rb"))
			prediction_2 = Linear_SVC.predict(vect_text)
			LR = joblib.load(open(os.path.join("resources/team7_logistic_regression_model.pkl"),"rb"))
			prediction_3 = LR.predict(vect_text)
			Complement_Naive_Bayes = joblib.load(open(os.path.join("resources/team7_complement_naive_bayes_model.pkl"), "rb"))
			prediction_4 = Complement_Naive_Bayes.predict(vect_text)

			if st.selectbox('Kernel SVC Model') and st.button('Classify'):
     					return prediction_1
			st.success("Text Categorized as: {}".format(prediction_1))
			if st.selectbox('Linear SVC Model') and st.button('Classify'):
     					return prediction_2
			st.success("Text Categorized as: {}".format(prediction_2))
			if st.selectbox('Logisitic Regression Model') and st.button('Classify'):
    					return prediction_3
			st.success("Text Categorized as: {}".format(prediction_3))
			if st.selectbox('Complement Naive Bayes Model') and st.button('Classify'):
      					return prediction_4
			st.success("Text Categorized as: {}".format(prediction_3))
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()


