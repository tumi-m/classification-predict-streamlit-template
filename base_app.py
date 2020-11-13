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
news_vectorizer = open("resources/tfidfvect.pkl","rb")
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
	#st.markdown(html_temp, unsafe_allow_html = True)


#image
	
image = Image.open('resources/imgs/Quiver.jpg')

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
options = ["Prediction", "Information", "About Team 7", "Contact Us"]
selection = st.sidebar.selectbox("Please selection an option", options)
st.markdown(
    """
<style>
.sidebar .sidebar-content {
	background-image: linear-gradient(#025246, #025246);
    color: white;
	font-size: 90px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Building out the "Information" page
if selection == "Information":
	st.info("General Information")
	# You can read a markdown file from supporting resources folder
	st.markdown("resources/info.md", unsafe_allow_html=True)

	st.subheader("Raw Twitter data and label")
	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
		st.write(raw[['sentiment', 'message']]) # will write the df to the page

#Building out the About Page
if selection == "About TEAM 7":
	#st.info("TEAM 7")
	st.write('TEAM 7 is a group of four members comprising of Thiyasize Khubeka, Warren Mnisi, Samuel Aina, and Tumelo Malebo')
	st.text("Visit our Contact Page and lets get in touch!")


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



# Building out the predication page
#def predict_age(tweet_id, sentiment):
    #input=np.array([[tweet_id, sentiment]]).astype(np.float64)
    #prediction = pred2.predict(input)
    #pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    #return int(prediction)

with st.beta_container():
	if selection == "Prediction":
	#st.info("Prediction with ML Models")
	# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		option = st.selectbox(
		'Which model would you like to use?',
		('Kernel SVC Model', 'Linear SVC Model', 'Logisitic Regression Model'))

		st.write('You selected the: ', option)


		if st.button("Classify"):
		# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
			
			
# Try loading in multiple models to give the user a choice
#predictor = joblib.load(open(os.path.join("resources/base_model.pkl"),"rb"))
#options = ["predictor1, predictor2, predictor3, predictor4"]
#selection = st.selectbox("Choose Option")

Kernel_SVC = joblib.load(open(os.path.join("resources/team7_kernel_svc_model.pkl"),"rb"))
Linear_SVC = joblib.load(open(os.path.join("resources/team7_linear_svc_model.pkl"),"rb"))
LR = joblib.load(open(os.path.join("resources/team7_logistic_regression_model.pkl"),"rb"))
#predictor4 = joblib.load(open(os.path.join("resources/pred2.pkl"),"rb"))

prediction_1 = Kernel_SVC.predict(vect_text)
prediction_2 = Linear_SVC.predict(vect_text)
prediction_3 = LR.predict(vect_text)
#prediction4 = predictor1.predict(vect_text)

#If statements for selecting different models
#if selection == 'SVC':
	#return prediction2
#elif selection == 'Logistic Regression':
	#return prediction3
# When model has successfully run, will print prediction
# You can use a dictionary or similar structure to make this output
# more human interpretable.
#st.success("Text Categorized as: {}".format(prediction))

if st.button("Classify the tweet") and st.selectbox("Kernel SVC"):
    output = prediction_1
st.success('Tweet classification {}'.format(output))
if st.button("Classify the tweet") and st.selectbox("Linear SVC"):
   	output = prediction_2
st.success('Tweet classification {}'.format(output))
if st.button("Classify the tweet") and st.selectbox("Logistic Regression"):
	output = prediction_3
st.success('Tweet classification {}'.format(output))


news_html ="""  
<div style="background-color:#80ff80; padding:10px >
<h2 style="color:white;text-align:center;"> This tweet links to factual news about climate change</h2>
</div>
"""
pro_html ="""
<div style="background-color:#F4D03F; padding:10px >
<h2 style="color:white;text-align:center;"> This tweet supports the belief of man-made climate climate</h2>
</div>
"""
neutral_html="""  
<div style="background-color:#F08080; padding:10px >
<h2 style="color:black ;text-align:center;">This tweet neither refutes or supports man-made climate change </h2>
</div>
"""
anti_html="""  
<div style="background-color:#F08080; padding:10px >
<h2 style="color:black ;text-align:center;"> This tweet does not believe in man-made climate change</h2>
</div>
"""


		
#elif st.button("Classify the tweet") and :
        	#output = prediction = predictor1.predict(vect_text)
        	#st.success('Tweet classification {}'.format(output))

		
    	#elif st.button("Classify the tweet"):
        	#output = prediction = predictor1.predict(vect_text)
        #st.success('Tweet classification {}'.format(output))

if output == 2:
    st.markdown(news_html,unsafe_allow_html=True)
elif output == 1:
   	st.markdown(pro_html,unsafe_allow_html=True)
elif output == 0:
    st.markdown(neutral_html,unsafe_allow_html=True)
elif output == 1:
    st.markdown(anti_html,unsafe_allow_html=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
