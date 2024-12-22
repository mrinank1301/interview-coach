import os
import openai
import streamlit as st
import PyPDF2
from docx import Document
import azure.cognitiveservices.speech as speechsdk
import time
from textblob import TextBlob
import threading
import datetime
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from textblob.download_corpora import download_all
import nltk

# Add the custom nltk_data path to the NLTK search paths
nltk.data.path.append('./nltk_data')
st.set_page_config(page_title="Azure OpenAI Interview Generator", layout="wide")
# Download corpora if not already downloaded
corpora_path = os.path.expanduser('~/.textblob/nltk_data')
if not os.path.exists(corpora_path):
    import streamlit as st
    st.info("Downloading TextBlob corpora. Please wait...")
    download_all()
    st.success("TextBlob corpora downloaded successfully!")

# Streamlit Interface

st.title("jobSpring AI Interview Generator")
st.markdown(
    """<style>
        .reportview-container {
            background: #f5f5f5;
        }
        h1 {
            color: #4a4a4a;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
        }
    </style>""",
    unsafe_allow_html=True
)

# Sidebar Configuration with Enhanced UI
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])
st.sidebar.markdown("---")

# Enhanced number of questions input
st.sidebar.subheader("Number of Questions")
num_questions = st.sidebar.number_input(
    "Enter the number of questions (1-50):",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
    help="Specify the exact number of questions you want to generate. Max: 50"
)

# Enhanced difficulty level selection
st.sidebar.subheader("Difficulty Level")
difficulty_options = ["Simple", "Moderate", "Difficult", "Custom"]
difficulty_level = st.sidebar.radio(
    "Select difficulty level:",
    options=difficulty_options,
    help="Choose a predefined difficulty level or create a custom one."
)

custom_difficulty = None
if difficulty_level == "Custom":
    custom_difficulty = st.sidebar.text_input(
        "Specify Custom Difficulty:",
        placeholder="E.g., Beginner, Advanced Behavioral",
        help="Define your custom difficulty level in a descriptive way."
    )

# Sidebar help section
st.sidebar.markdown("---")
show_help = st.sidebar.checkbox("Show Help")

if show_help:
    st.sidebar.info(
        """
        **Instructions:**
        - Upload a resume file in PDF, DOCX, or TXT format.
        - Specify the number of questions and difficulty level.
        - If using 'Custom Difficulty,' describe it clearly.
        - Click 'Generate Interview Questions' to proceed.
        """
    )

# OpenAI API keys
openai_api_key = "22Mj7xKp5fPvOKQDZ54xncvwHCUUt27nPBhmgI89k60HJ3do1kgTJQQJ99ALACYeBjFXJ3w3AAABACOGOy3V"
openai_endpoint_url = "https://jobspringai.openai.azure.com/"
openai_deployment_name = "gpt-35-turbo"

# Azure Speech API keys
speech_api_key = "2XJ6aPd30utg3LgzQtxd3peUvYK92O42pt1zZySHWSXUB9OLyVS5JQQJ99ALACYeBjFXJ3w3AAAYACOGufOC"
speech_endpoint = "https://eastus.api.cognitive.microsoft.com/"

# Function to extract text from uploaded files
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function for advanced sentiment analysis using GPT-4
def analyze_sentiment(text):
    openai.api_key = openai_api_key
    try:
        response = openai.Completion.create(
            engine=openai_deployment_name,
            prompt=f"Analyze the sentiment of the following text and provide a detailed analysis (emotion, tone, intensity, polarity, etc.):\n\n{text}",
            max_tokens=150,
            temperature=0.7
        )
        sentiment_text = response.choices[0].text.strip()

        # Let's assume the response contains emotional tone (like "positive", "negative") and polarity
        # Parsing the response into a structured dictionary
        sentiment_details = {
            "emotion": "Neutral",  # Default value
            "polarity": 0.0,       # Default value
            "intensity": "Low"     # Default value
        }
        
        # Parsing the text response to detect sentiment attributes
        if "positive" in sentiment_text.lower():
            sentiment_details["emotion"] = "Positive"
            sentiment_details["polarity"] = 1.0
            sentiment_details["intensity"] = "High"
        elif "negative" in sentiment_text.lower():
            sentiment_details["emotion"] = "Negative"
            sentiment_details["polarity"] = -1.0
            sentiment_details["intensity"] = "High"
        else:
            sentiment_details["emotion"] = "Neutral"
            sentiment_details["polarity"] = 0.0
            sentiment_details["intensity"] = "Low"

        return sentiment_details

    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return None


# Function to extract key phrases (Keywords) from the response
def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Function to calculate response quality score (based on sentiment and keywords)
def calculate_quality_score(sentiment_details, keywords):
    # Base score: polarity + keyword count
    base_score = sentiment_details["polarity"] * 10 + len(keywords)
    
    # Adjust score based on emotion
    if sentiment_details["emotion"] == "Positive":
        base_score += 3
    elif sentiment_details["emotion"] == "Negative":
        base_score -= 3
    
    # Normalize score to be between 0 and 10
    score = max(0, min(base_score, 10))
    return score

# Real-time interview interaction with a slight pause for user to prepare to speak
def ask_and_respond(question, speech_recognizer, speech_synthesizer, duration):
    # Speak the question first
    speech_synthesizer.speak_text_async(question)
    st.write(f"**Interviewer asks:** {question}")

    # Give a brief pause before starting to listen to user input
    time.sleep(5)  # Adding a small delay for the user to be ready

    user_answer = None
    sentiment = None
    start_time = time.time()

    # Now, let's start listening for the user's response
    with st.spinner("Listening..."):
        while time.time() - start_time < duration and user_answer is None:
            result = speech_recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                user_answer = result.text
                sentiment = analyze_sentiment(user_answer)
                st.write(f"**Your Answer:** {user_answer}")
                st.write(f"**Sentiment:** {sentiment}")
                return user_answer, sentiment

    # If no answer was captured within the given time, return a default response
    if user_answer is None:
        st.warning("No valid input detected. Moving to the next question.")
        return "No answer recorded", "Neutral"

    return user_answer, sentiment
  


# Function to compare resume text with job description
def compare_with_job_description(resume_text, job_description):
    resume_blob = TextBlob(resume_text)
    job_blob = TextBlob(job_description)
    
    resume_words = set(resume_blob.words)
    job_words = set(job_blob.words)
    
    matching_words = resume_words.intersection(job_words)
    match_percentage = (len(matching_words) / len(job_words)) * 100 if job_words else 0
    
    return match_percentage, list(matching_words)

# Function to calculate time taken per answer
def calculate_time_taken(start_time):
    end_time = time.time()
    time_taken = end_time - start_time
    return str(datetime.timedelta(seconds=int(time_taken)))

# Resume Skills Extraction
def extract_skills(resume_text):
    skills_keywords = ["Python", "Java", "JavaScript", "SQL", "Data Science", "Machine Learning", "Teamwork", "Leadership", "Communication", "Problem Solving", "Project Management", "Git"]
    skills_found = [skill for skill in skills_keywords if skill.lower() in resume_text.lower()]
    return skills_found

# Function to generate interview tips based on sentiment and score
def generate_interview_tips(sentiment, score):
    tips = []
    if sentiment["emotion"] == "Positive":
        tips.append("Maintain your positive tone.")
    elif sentiment["emotion"] == "Negative":
        tips.append("Try to be more positive in your responses.")
    
    if score < 5:
        tips.append("Provide more detailed answers.")
    elif score >= 5:
        tips.append("Keep up the good work and continue to be concise.")
    
    return tips

# Main logic
if uploaded_file is not None:
    with st.spinner("Processing the uploaded file..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    if text.strip():
        st.subheader("Extracted Text from Resume")
        with st.expander("Show Extracted Text"):
            st.write(text)

        # Job description input
        st.subheader("Compare with Job Description")
        job_description = st.text_area("Paste the job description here:")
        if st.button("Analyze Match"):
            if job_description.strip():
                match_percentage, matching_words = compare_with_job_description(text, job_description)
                st.write(f"**Match Percentage:** {match_percentage:.2f}%")
                st.write(f"**Matching Keywords:** {', '.join(matching_words)}")
            else:
                st.warning("Please enter a job description.")

        # Skills Detection
        skills = extract_skills(text)
        if skills:
            st.subheader("Key Skills from Resume")
            st.write(f"**Skills Detected:** {', '.join(skills)}")
        else:
            st.write("No specific skills detected in the resume.")

        # Generate Questions Button
        # Inside the "Generate Interview Questions" button logic

# Inside the "Generate Interview Questions" button logic

if st.button("Generate Interview Questions"):
    openai.api_type = "azure"
    openai.api_base = openai_endpoint_url
    openai.api_version = "2024-05-01-preview"
    openai.api_key = openai_api_key

    try:
        chat_prompt = [
            {"role": "system", "content": f"You are a professional interviewer conducting a real-time, natural, and engaging conversation. Generate {num_questions} {difficulty_level} interview questions based on the following text:"},
            {"role": "user", "content": text}
        ]

        completion = openai.ChatCompletion.create(
            engine=openai_deployment_name,
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7
        )

        response = completion.choices[0].message["content"]
        st.subheader("Generated Interview Questions")
        st.write(response)

        # Speech SDK configuration
        speech_config = speechsdk.SpeechConfig(subscription=speech_api_key, region="eastus")
        speech_config.speech_recognition_language = "en-US"
        speech_config.speech_synthesis_voice_name = "en-US-JessaNeural"

        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

        question_list = response.split("\n")
        user_responses = []  # Initialize the list before the loop

        progress_bar = st.progress(0)
        total_questions = len(question_list)

        start_time = time.time()
        for idx, question in enumerate(question_list):
            if question.strip():
                st.write(f"**Question {idx + 1}:** {question}")
                answer, sentiment = ask_and_respond(question, speech_recognizer, speech_synthesizer, duration=30)
                sentiment_details = analyze_sentiment(answer)
                keywords = extract_keywords(answer)
                score = calculate_quality_score(sentiment_details, keywords)
                time_taken = calculate_time_taken(start_time)

                # Store response in the user_responses list
                user_responses.append({
                    "question": question,
                    "answer": answer,
                    "sentiment": sentiment_details,
                    "keywords": keywords,
                    "score": score,
                    "time_taken": time_taken
                })

                # Update progress bar after each question
                progress = int((idx + 1) / total_questions * 100)
                progress_bar.progress(progress)

        # Ensure interview data is available before trying to display or download
        if user_responses:
            st.subheader("Interview Summary")
            for response in user_responses:
                st.write(f"**Question:** {response['question']}")
                st.write(f"**Answer:** {response['answer']}")
                st.write(f"**Sentiment:** {response['sentiment']}")
                st.write(f"**Keywords:** {', '.join(response['keywords'])}")
                st.write(f"**Quality Score:** {response['score']}/10")
                st.write(f"**Time Taken:** {response['time_taken']}")
                st.write("-" * 50)

            # Provide the option to download the interview results as CSV or JSON
            interview_data = pd.DataFrame(user_responses)
            st.subheader("Download Interview Results")
            csv = interview_data.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="interview_results.csv",
                mime="text/csv"
            )

            # Optionally add download for JSON
            json_data = interview_data.to_json(orient="records", lines=True)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name="interview_results.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
