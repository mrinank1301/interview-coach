import os
import textblob

def download_corpora():
    print("Downloading TextBlob corpora...")
    textblob.download_corpora()
    print("Download complete!")