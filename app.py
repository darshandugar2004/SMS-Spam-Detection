import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data if not already downloaded
# This is crucial for the transform_text function to work
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
    """
    Preprocesses the input text by:
    1. Lowercasing
    2. Tokenization
    3. Removing non-alphanumeric characters
    4. Removing stopwords and punctuation
    5. Stemming using Porter Stemmer
    """
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    # 1. Removing non-alphanumeric characters
    for i in text:
        if i.isalnum():  # Check if the character is alphanumeric
            y.append(i)

    text = y[:]  # Create a copy of y to update 'text'
    y.clear()  # Clear y for the next step

    # 2. Removing stopwords and punctuation
    stop_words_english = stopwords.words('english')  # Get English stopwords
    for i in text:
        if i not in stop_words_english and i not in string.punctuation:
            y.append(i)

    text = y[:]  # Update 'text' with cleaned words
    y.clear()  # Clear y for the next step

    # 3. Stemming
    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)  # Join the stemmed words back into a string

# Load the pre-trained TF-IDF vectorizer and the model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found. "
             "Please ensure these files are in the same directory as the Streamlit app.")
    st.stop() # Stop the app if models are not found

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Detector", layout="centered")

st.title("SMS Spam Detector")
st.markdown("""
    Enter an SMS message below to check if it's spam or not.
    This application uses a pre-trained machine learning model to classify messages.
""")

# Text area for user input
sms_input = st.text_area("Enter the SMS message here:", height=150)

# Prediction button
if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter an SMS message to predict.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(sms_input)

        # 2. Vectorize the preprocessed text
        # The input to transform needs to be a list/array-like
        vector_input = tfidf.transform([transformed_sms])

        # 3. Make prediction
        prediction = model.predict(vector_input)[0]

        # 4. Display the result
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("🚨 SPAM! This message is likely spam.")
            st.markdown("---")
            st.write("Be cautious with this message. It might contain suspicious links or requests.")
        else:
            st.success("✅ NOT SPAM! This message appears to be legitimate.")
            st.markdown("---")
            st.write("This message is likely safe. However, always be vigilant with unknown senders.")

st.markdown("---")
st.caption("Developed using Streamlit, NLTK, and Scikit-learn.")
