import nltk
import textblob
from textblob import download_corpora

def download_nltk_data():
    try:
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('brown')
        nltk.download('wordnet')
        # Download TextBlob corpora
        download_corpora()
        print("Successfully downloaded all required corpora")
    except Exception as e:
        print(f"Error downloading corpora: {e}")

if __name__ == "__main__":
    download_nltk_data()