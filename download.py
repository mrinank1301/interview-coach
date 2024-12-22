import nltk
import streamlit as st
from textblob import TextBlob

@st.cache_resource
def download_corpora():
    """
    Downloads required NLTK corpora for TextBlob.
    This function is cached so it only runs once per session.
    """
    try:
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('brown')
        nltk.download('wordnet')
        nltk.download('conll2000')
        st.success("Successfully downloaded TextBlob dependencies!")
        return True
    except Exception as e:
        st.error(f"Error downloading corpora: {str(e)}")
        return False