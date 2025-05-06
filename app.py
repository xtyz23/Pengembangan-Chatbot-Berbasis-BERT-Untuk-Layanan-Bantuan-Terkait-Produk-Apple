import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import re
import json
import nltk
from nltk.corpus import stopwords
from safetensors.torch import load_file
import pandas as pd
from Levenshtein import ratio
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
import csv
import io

# Page configuration
st.set_page_config(page_title="Digital Support Assistant", layout="wide")

# Enhanced Custom CSS for styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #000000 0%, #1a1a1a 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #007AFF 0%, #5856D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Chat Message Styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .chat-message.bot {
        background: white;
        border-bottom-left-radius: 5px;
        margin-right: auto;
    }
    
    .chat-message.user::before,
    .chat-message.bot::before {
        content: '';
        position: absolute;
        bottom: -10px;
        width: 20px;
        height: 20px;
    }
    
    .message-content {
        line-height: 1.5;
    }
    
    .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        text-align: right;
    }

    /* Batch Processing Styles */
    .file-uploader {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #007AFF;
    }
    
    .result-question {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    .result-answer {
        color: #555;
    }
    
    .result-confidence {
        font-size: 0.8rem;
        color: #888;
        text-align: right;
        margin-top: 0.5rem;
    }
    
    /* Input Container Styles */
    .chat-input-container {
        display: flex;
        gap: 1rem;
        align-items: center;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1.5rem;
        background: white;
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0,0,0,0.1);
        z-index: 100;
    }
    
    /* Input Field Styles */
    .stTextInput input {
        background-color: white !important;
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }
    
    .stTextInput input:focus {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 2px rgba(0,122,255,0.2) !important;
    }
    /* Button Styles */
    .stButton button {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }
    /* Hide Streamlit Components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Scrollbar Styles */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #007AFF;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5856D6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm your Apple Support Assistant. How can I help you today?"}]
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Load stopwords
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model(model_name):
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=837)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    if model_name == "Model 1(3e-5,8)":
        weights = load_file('model1.safetensors')
    elif model_name == "Model 2(3e-5,16)":
        weights = load_file('model3.safetensors')
    elif model_name == "Model 3(3e-5,32)":
        weights = load_file('model3.safetensors')
    elif model_name == "Model 4(4e-5,8)":
        weights = load_file('model4.safetensors')
    elif model_name == "Model 5(4e-5,16)":
        weights = load_file('model5.safetensors')
    elif model_name == "Model 6(4e-5,32)":
        weights = load_file('model6.safetensors')
    model.load_state_dict(weights)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Sidebar for task and model selection
st.sidebar.title("Settings")
task = st.sidebar.radio("Choose a task", ["Chat with Assistant", "Batch Question Processing"])
model_name = st.sidebar.radio("Choose a model", ["Model 1(3e-5,8)", "Model 2(3e-5,16)", "Model 3(3e-5,32)", "Model 4(4e-5,8)", "Model 5(4e-5,16)", "Model 6(4e-5,32)"])

# Load model and tokenizer based on selection
try:
    model, tokenizer = load_model(model_name)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

@st.cache_data
def load_responses():
    df = pd.read_csv('augmented_data.csv')  # Load the CSV

    responses = {}
    patterns_dict = {}

    for _, row in df.iterrows():
        tag = row['tag']
        pattern = row['patterns']
        response = row['responses']

        # Store response only for the first time a tag appears
        if tag not in responses:
            responses[tag] = response

        # Store patterns grouped by tag
        if tag in patterns_dict:
            patterns_dict[tag].append(pattern)
        else:
            patterns_dict[tag] = [pattern]

    return responses, patterns_dict

label_to_response, patterns_dict = load_responses()

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

from Levenshtein import ratio
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

DEVICE_KEYWORDS = {"iphone", "ipad", "mac", "apple watch", "macbook", "airpods", "apple tv", "homepod", "imac"}
CONFIDENCE_THRESHOLD = 0.40
WORD_MATCH_THRESHOLD = 2  # Minimum matching words required
SOFT_MATCH_WORDS = {"using", "use", "possible"}


def extract_device(text):
    words = preprocess_text(text).split()
    for word in words:
        if word in DEVICE_KEYWORDS:
            return word
    return None  # No device found

lemmatizer = WordNetLemmatizer()

def is_similar(word, word_list, threshold=0.8):
    """
    Finds the closest matching word using Levenshtein ratio and lemmatization.
    """
    # Normalize input word (convert verb forms to base form)
    word = lemmatizer.lemmatize(word, pos="v")  

    # Find the best match in the list (also lemmatized)
    best_match = max(word_list, key=lambda w: ratio(word, lemmatizer.lemmatize(w, pos="v")), default=None)
    
    # Return best match only if it meets the similarity threshold
    return best_match if best_match and ratio(word, best_match) >= threshold else None

def word_match_count(user_input, label_patterns):
    user_words = set(preprocess_text(user_input).split()) - stop_words   
    max_match_score = 0   
    match_scores = {}  

    for tag, patterns in label_patterns.items():
        for pattern in patterns:
            pattern_words = set(preprocess_text(pattern).split()) - stop_words

            # Apply Levenshtein with Lemmatization
            matched_words = [is_similar(word, pattern_words) for word in user_words]
            matched_words = [w for w in matched_words if w]  # Remove None values

            # Calculate total match score (with weighted words)
            match_score = sum(0.5 if word in SOFT_MATCH_WORDS else 1 for word in matched_words)

            if match_score >= WORD_MATCH_THRESHOLD:
                match_scores[tag] = max(match_scores.get(tag, 0), match_score)
                max_match_score = max(max_match_score, match_score)

    best_labels = [tag for tag, score in match_scores.items() if score == max_match_score]
    return best_labels  


def get_response(input_text):
    preprocessed_text = preprocess_text(input_text)
    
    # Step 1: Extract Device from User Input
    device = extract_device(input_text)
    # Step 2: Filter Labels by Device
    if device:
        filtered_labels = {
            tag: patterns for tag, patterns in patterns_dict.items()
            if any(device in pattern.lower() for pattern in patterns)  # Check patterns for device
        }
        
        if not filtered_labels:
            return "Sorry, I couldn't find a relevant response for your device.", 0.0
    else:
        filtered_labels = patterns_dict  # No device found, use all labels
    # Step 3: Apply Word Matching to Find Best Labels
    best_labels = word_match_count(input_text, filtered_labels)
    # If no best labels found, return a fallback response
    if not best_labels:
        return "I'm not sure about that. Can you rephrase or ask something else?", 0.0
    # Step 4: Restrict Model Prediction to Only the Best Labels
    best_label_indices = [list(label_to_response.keys()).index(tag) for tag in best_labels]
    
    # Tokenize input
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    # Step 5: Get Model Prediction Only for Best Labels
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Keep only logits corresponding to best labels
        filtered_logits = logits[:, best_label_indices]
        probabilities = F.softmax(filtered_logits, dim=1)
        confidence, selected_index = torch.max(probabilities, dim=1)
    confidence_value = confidence.item()
    predicted_label = best_labels[selected_index.item()]  # Pick from best labels
    # Step 6: Apply Confidence Threshold
    if confidence_value < CONFIDENCE_THRESHOLD:
        return "I'm not sure about that. Can you rephrase or ask something else?", confidence_value
    return label_to_response[predicted_label], confidence_value

def process_input():
    if st.session_state.user_input.strip():
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
        
        bot_response, confidence_value = get_response(st.session_state.user_input)
        confidence_display = f"(Confidence: {confidence_value:.2f})"
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response, "confidence": confidence_display})
        st.session_state.user_input = ""

def process_batch_file(file):
    questions = []
    labels = []
    has_labels = False
    
    # Determine file type and read accordingly
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        if 'question' in df.columns:
            questions = df['question'].tolist()
            # Check if label column exists
            if 'label' in df.columns:
                labels = df['label'].tolist()
                has_labels = True
        else:
            questions = df.iloc[:, 0].tolist()  # Assume first column contains questions
            # Check if there's a second column for labels
            if df.shape[1] > 1:
                labels = df.iloc[:, 1].tolist()
                has_labels = True
    elif file.name.endswith('.txt'):
        content = file.getvalue().decode('utf-8')
        questions = [q.strip() for q in content.split('\n') if q.strip()]
    else:
        st.error("Unsupported file format. Please upload a CSV or TXT file.")
        return [], has_labels, 0.0
    
    # Process each question
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for i, question in enumerate(questions):
        if question:  # Skip empty lines
            preprocessed_text = preprocess_text(question)
            
            # Get the model's prediction
            answer, confidence = get_response(question)
            
            # Get the predicted label
            # We need to find which label corresponds to this answer
            predicted_label = None
            for label, response in label_to_response.items():
                if response == answer:
                    predicted_label = label
                    break
            
            result = {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "predicted_label": predicted_label
            }
            
            # If we have true labels, calculate accuracy
            if has_labels and i < len(labels):
                true_label = str(labels[i])
                result["true_label"] = true_label
                
                # Convert label types to match (both should be strings)
                if str(predicted_label) == true_label:
                    correct_predictions += 1
                total_predictions += 1
            
            results.append(result)
    
    # Calculate accuracy if we have labels
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    return results, has_labels, accuracy

def download_results_as_csv():
    if not st.session_state.batch_results:
        return
    
    # Create a DataFrame from the results
    df = pd.DataFrame(st.session_state.batch_results)
    
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="support_assistant_results.csv",
        mime="text/csv"
    )

# Main UI
st.markdown('<div class="main-header"><h1 class="header-title">Apple Support Assistant</h1></div>', unsafe_allow_html=True)

# Display the selected task
if task == "Chat with Assistant":
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role_class = "user" if message["role"] == "user" else "bot"
            confidence_text = message.get("confidence", "")  # Get confidence if available
            
            st.markdown(f"""
            <div class="chat-message {role_class}">
                <div class="message-content">{message["content"]}</div>
                <div class="timestamp">{'You' if role_class == 'user' else 'Support Assistant'} • Just now {confidence_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])
        with col1:
            st.text_input("Type a Message...", key="user_input", placeholder="Type your message here...", label_visibility="collapsed")
        with col2:
            if st.button("Send", on_click=process_input):
                pass
        st.markdown("</div>", unsafe_allow_html=True)

elif task == "Batch Question Processing":
    st.markdown("<div class='file-uploader'>", unsafe_allow_html=True)
    st.subheader("Batch Question Processing")
    st.write("Upload a file with multiple questions to get answers all at once.")
    
    upload_help = """
    **Supported file formats:**
    - CSV file with a 'question' column and an optional 'tag' column
    - TXT file with one question per line
    
    **Example CSV format:**
    ```
    question,label
    How do I reset my iPhone?
    How to update macOS?
    My AirPods won't connect
    ```
    
    **Note:** Including a 'tag' column allows the system to calculate accuracy based on known correct labels.
    """
    st.markdown(upload_help)
    
    uploaded_file = st.file_uploader("Upload questions file", type=["csv", "txt"])
    
    accuracy_displayed = False
    
    if uploaded_file is not None:
        if st.button("Process Questions"):
            with st.spinner("Processing questions..."):
                results, has_labels, accuracy = process_batch_file(uploaded_file)
                if results:
                    st.session_state.batch_results = results
                    st.session_state.has_labels = has_labels
                    st.session_state.accuracy = accuracy
                    accuracy_displayed = True
                    st.success(f"Processed {len(results)} questions!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display accuracy if available
    if 'has_labels' in st.session_state and st.session_state.has_labels:
        st.markdown(f"""
        <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #007AFF;">
            <h3 style="margin-top: 0; color: #007AFF;">Model Accuracy</h3>
            <p style="font-size: 18px; font-weight: bold;">Accuracy: {st.session_state.accuracy:.2f}%</p>
            <p>Based on {len(st.session_state.batch_results)} questions with known labels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results if available
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        st.subheader("Results")
        
        # Download button for results
        download_results_as_csv()
        
        # Display each question-answer pair
        for i, result in enumerate(st.session_state.batch_results):
            # Determine if prediction was correct
            correct_class = ""
            label_info = ""
            
            if 'true_label' in result and 'predicted_label' in result:
                is_correct = str(result['predicted_label']) == str(result['true_label'])
                correct_class = "correct-prediction" if is_correct else "incorrect-prediction"
            
            st.markdown(f"""
            <div class="result-card {correct_class}">
                <div class="result-question">Q: {result['question']}</div>
                <div class="result-answer">A: {result['answer']}</div>
                {label_info}
                Confidence: {result['confidence']:.2f}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Add additional CSS for correct/incorrect predictions
st.markdown("""
<style>
    .correct-prediction {
        color: #28a745;
        font-weight: bold;
    }
    
    .incorrect-prediction {
        color: #dc3545;
        font-weight: bold;
    }
    
    .result-card.correct-prediction {
        border-left: 4px solid #28a745;
    }
    
    .result-card.incorrect-prediction {
        border-left: 4px solid #dc3545;
    }
    
    .result-labels {
        font-size: 0.9rem;
        margin: 0.5rem 0;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)