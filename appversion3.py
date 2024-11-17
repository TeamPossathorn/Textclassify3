import streamlit as st
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('model.joblib')
# Define stopwords
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }

    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features

def parse(text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    
    try:
        # Assuming 'model' is defined and loaded elsewhere in your code
        predictions = model.predict([features])[0]
        return predictions
    except NameError as e:
        st.error("Model is not defined. Please ensure the model is loaded correctly.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Streamlit UI
st.title("Named Entity Recognition (NER) Prediction App")
st.write("กรุณาป้อนข้อมูลข้อความที่ต้องการให้โมเดลทำการตรวจจับ Named Entities")

# Text input from user
user_input = st.text_area("ป้อนข้อความที่นี่", height=150)

if st.button("Parse"):
    if user_input:
        results = parse(user_input)
        
        if results is not None:  # Check if results were returned successfully
            # Display input and output in a horizontal layout (as rows)
            st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line for separation
            st.write("### Input:")
            st.write(user_input)

            st.write("### Predictions:")
            st.text(results)
        st.warning("Please enter some text to parse.")
