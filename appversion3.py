import streamlit as st
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np

# Load the model
model = joblib.load("model(1).joblib")

# Define stopwords and feature extraction function
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace() if isinstance(word, str) else False,
        "word.is_stopword()": word in stopwords if isinstance(word, str) else False,
        "word.isdigit()": word.isdigit() if isinstance(word, str) else False,
        "word.islen5": word.isdigit() and len(word) == 5 if isinstance(word, str) else False
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace() if isinstance(prevword, str) else False,
            "-1.word.is_stopword()": prevword in stopwords if isinstance(prevword, str) else False,
            "-1.word.isdigit()": prevword.isdigit() if isinstance(prevword, str) else False
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace() if isinstance(nextword, str) else False,
            "+1.word.is_stopword()": nextword in stopwords if isinstance(nextword, str) else False,
            "+1.word.isdigit()": nextword.isdigit() if isinstance(nextword, str) else False
        })
    else:
        features["EOS"] = True
    return features

# Function to run the model and display results
def run_model(tokens):
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    return model.predict([features])[0]

# Display entities with color coding
def display_results(tokens, correct_tags, predicted_tags, typo_indices):
    result_html = ''
    for i, (token, correct_tag, predicted_tag) in enumerate(zip(tokens, correct_tags, predicted_tags)):
        color = 'black' if correct_tag == predicted_tag else 'red'
        if i in typo_indices:
            typo_idx = typo_indices[i]
            if typo_idx is not None and typo_idx < len(token):
                token = token[:typo_idx] + f'<span style="color:cyan">{token[typo_idx]}</span>' + token[typo_idx+1:]
        result_html += f'<span style="color:{color}">{token} - {predicted_tag}</span> '
    st.markdown(result_html, unsafe_allow_html=True)

def plot_entity_distribution(predicted_tags):
    entity_counts = Counter(predicted_tags)
    
    # Check if entity_counts has any items before unpacking
    if entity_counts:
        labels, values = zip(*entity_counts.items())
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.title('Entity Distribution in Predictions')
        st.pyplot(fig)
    else:
        st.write("No predicted tags available for entity distribution.")


# Plot cumulative confusion matrix
def plot_cumulative_confusion_matrix():
    labels = sorted(set(st.session_state['all_true_tags']) | set(st.session_state['all_predicted_tags']))
    cm = confusion_matrix(st.session_state['all_true_tags'], st.session_state['all_predicted_tags'], labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Cumulative Confusion Matrix")
    st.pyplot(fig)

# Common function to update and display results
def update_display(tokens, correct_tags):
    predicted_tags = run_model(tokens)
    display_results(tokens, correct_tags, predicted_tags, st.session_state['typo_indices'])
    #plot_entity_distribution(predicted_tags)
    # Accumulate results for cumulative confusion matrix
    st.session_state['all_true_tags'].extend(correct_tags)
    st.session_state['all_predicted_tags'].extend(predicted_tags)
    #plot_cumulative_confusion_matrix()

# Initialize session state variables if they do not exist
if 'modified_tokens' not in st.session_state:
    st.session_state['modified_tokens'] = []
    st.session_state['typo_indices'] = {}

# Define typo introduction function
def introduce_realistic_typos(tokens):
    typo_indices = {}
    for idx in random.sample(range(len(tokens)), max(1, len(tokens) // 2)):
        token = tokens[idx]
        if len(token) == 0:
            continue
        typo_type = random.choice(['substitute', 'omit', 'transpose', 'duplicate'])
        chars = list(token)
        i = random.randint(0, len(chars) - 1)
        if typo_type == 'substitute':
            chars[i] = chr((ord(chars[i]) & ~0xFF) + random.randint(0, 255))
        elif typo_type == 'omit':
            chars.pop(i)
        elif typo_type == 'transpose' and len(chars) > 1:
            if i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif typo_type == 'duplicate':
            chars.insert(i, chars[i])
        tokens[idx] = ''.join(chars)
        typo_indices[idx] = i
    return tokens, typo_indices

# Initialize session state variables
if 'original_tokens' not in st.session_state:
    st.session_state['original_tokens'] = []
    st.session_state['original_correct_tags'] = []
    st.session_state['modified_tokens'] = []
    st.session_state['modified_correct_tags'] = []
    st.session_state['typo_indices'] = {}
if 'all_true_tags' not in st.session_state:
    st.session_state['all_true_tags'] = []
if 'all_predicted_tags' not in st.session_state:
    st.session_state['all_predicted_tags'] = []

# Divide layout into three columns in one row
col1, spacer, col2 = st.columns([1.25,0.5, 2])

# Column 1: Input Section
with col1:
    # Apply the custom class to each text input
    st.markdown("<p style='font-size:12px;'>Name</p>", unsafe_allow_html=True)
    name_text = st.text_input("", key="name", placeholder="Enter Name", help="Your name")

    st.markdown("<p style='font-size:12px;'>Street Address</p>", unsafe_allow_html=True)
    street_text = st.text_input("", key="street_address", placeholder="Enter Street Address", help="Your street address")

    st.markdown("<p style='font-size:12px;'>Subdistrict (Tambon)</p>", unsafe_allow_html=True)
    subdistrict_text = st.text_input("", key="subdistrict", placeholder="Enter Subdistrict", help="Your subdistrict")

    st.markdown("<p style='font-size:12px;'>District (Amphoe)</p>", unsafe_allow_html=True)
    district_text = st.text_input("", key="district", placeholder="Enter District", help="Your district")

    st.markdown("<p style='font-size:12px;'>Province</p>", unsafe_allow_html=True)
    province_text = st.text_input("", key="province", placeholder="Enter Province", help="Your province")

    st.markdown("<p style='font-size:12px;'>Postal Code</p>", unsafe_allow_html=True)
    postal_code_text = st.text_input("", key="postal_code", placeholder="Enter Postal Code", help="Your postal code")


    if st.button("Run Model"):
        full_address = f"{name_text} {street_text} {subdistrict_text} {district_text} {province_text} {postal_code_text}"
        tokens = full_address.split()
        correct_tags = (
            ['O'] * len(name_text.split()) +
            ['ADDR'] * len(street_text.split()) +
            ['LOC'] * len(subdistrict_text.split()) +
            ['LOC'] * len(district_text.split()) +
            ['LOC'] * len(province_text.split()) +
            ['POST'] * len(postal_code_text.split())
        )
        if len(tokens) != len(correct_tags):
            st.error("Error: Number of tokens and tags do not match.")
        else:
            st.session_state.update({
                'original_tokens': tokens,
                'original_correct_tags': correct_tags,
                'modified_tokens': tokens.copy(),
                'modified_correct_tags': correct_tags.copy(),
                'typo_indices': {}
            })
            update_display(tokens, correct_tags)

    if st.button("Scramble"):
        combined = list(zip(st.session_state['modified_tokens'], st.session_state['modified_correct_tags']))
        random.shuffle(combined)
        st.session_state['modified_tokens'], st.session_state['modified_correct_tags'] = zip(*combined)
        st.session_state['typo_indices'] = {}
        update_display(st.session_state['modified_tokens'], st.session_state['modified_correct_tags'])

    if st.button("Simulate Typo"):
        st.session_state['modified_tokens'], st.session_state['typo_indices'] = introduce_realistic_typos(st.session_state['modified_tokens'].copy())
        update_display(st.session_state['modified_tokens'], st.session_state['modified_correct_tags'])

# Column 2: Named Entity Distribution
with col2:
    # Row 1 in Column 2
    with st.container():
        st.markdown("### Entity Distribution")
        # Entity distribution plot or content goes here
        plot_entity_distribution(st.session_state.get("all_predicted_tags", []))

    # Row 2 in Column 2
    with st.container():
        st.write("### Cumulative Confusion Matrix")
    if st.session_state.get("all_true_tags") and st.session_state.get("all_predicted_tags"):
        plot_cumulative_confusion_matrix()
