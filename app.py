import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess
import sys
from PIL import Image

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Download stopwords
@st.cache_resource
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_stopwords()

def preprocess_text(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def check_model_exists():
    return os.path.exists('fake_news_model.h5') and os.path.exists('tokenizer.pkl')

@st.cache_resource
def load_model_and_tokenizer():
    if not check_model_exists():
        return None, None
    model = load_model('fake_news_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def train_model():
    with st.spinner("🔄 Training model... This will take 5-15 minutes..."):
        try:
            if not os.path.exists('train.csv'):
                st.error("❌ train.csv not found!")
                st.markdown("[📥 Download Dataset](https://www.kaggle.com/c/fake-news/data)")
                return False
            
            process = subprocess.Popen(
                [sys.executable, 'train.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            output_placeholder = st.empty()
            output_text = ""
            
            for line in process.stdout:
                output_text += line
                output_placeholder.text_area("Training Progress:", output_text, height=400)
            
            process.wait()
            
            if process.returncode == 0:
                st.success("✅ Training complete!")
                st.balloons()
                return True
            else:
                st.error(f"❌ Training failed: {process.stderr.read()}")
                return False
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False

def main():
    # Header
    st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">📰 Fake News Detection System</h1><p style="color: white; margin: 0;">Powered by LSTM Deep Learning</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard")
        
        if check_model_exists():
            st.success("✅ Model Ready")
            
            # Show visualizations if they exist
            if os.path.exists('visualizations'):
                st.markdown("### 📈 Visualizations")
                viz_files = {
                    'wordclouds.png': 'Word Clouds',
                    'training_history.png': 'Training History',
                    'confusion_matrix.png': 'Confusion Matrix'
                }
                
                for filename, title in viz_files.items():
                    filepath = f'visualizations/{filename}'
                    if os.path.exists(filepath):
                        with st.expander(f"📊 {title}"):
                            image = Image.open(filepath)
                            st.image(image, use_container_width=True)
        else:
            st.warning("⚠️ Model Not Trained")
        
        st.markdown("---")
        st.markdown("### ℹ️ About LSTM")
        st.info("""
        **Why LSTM?**
        - Captures long-term dependencies
        - Understands context in text
        - Effective for sequential data
        - Prevents vanishing gradients
        """)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None:
        # Training section
        st.warning("⚠️ **Model not found!** Train the model first.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Quick Setup:
            1. Ensure \`train.csv\` is in the project folder
            2. Click "Train Model" button
            3. Wait 5-15 minutes
            4. Start predicting!
            """)
            st.markdown("[📥 Download Dataset from Kaggle](https://www.kaggle.com/c/fake-news/data)")
        
        with col2:
            st.markdown("### ")
            if st.button("🎯 Train Model", type="primary"):
                if train_model():
                    st.rerun()
        
        st.markdown("---")
        st.info("💡 The model only needs to be trained once!")
        
    else:
        # Prediction interface
        st.success("✅ Model loaded and ready!")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Examples", "📖 How It Works"])
        
        with tab1:
            st.subheader("Enter News Article")
            news_text = st.text_area(
                "Paste news text here:",
                placeholder="Enter or paste a news article...",
                height=200
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                predict_btn = st.button("🔍 Analyze News", type="primary")
            with col2:
                if st.button("🗑️ Clear"):
                    st.rerun()
            
            if predict_btn and news_text.strip():
                with st.spinner("🤔 Analyzing..."):
                    cleaned = preprocess_text(news_text)
                    sequence = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(sequence, maxlen=300, padding='post')
                    prediction = model.predict(padded, verbose=0)[0][0]
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction > 0.5:
                            st.success("### ✅ REAL NEWS")
                        else:
                            st.error("### ❌ FAKE NEWS")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    with col3:
                        st.metric("Prediction Score", f"{prediction:.4f}")
                    
                    st.progress(float(prediction))
                    
                    with st.expander("🔍 Preprocessed Text"):
                        st.text(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)
            
            elif predict_btn:
                st.warning("⚠️ Please enter text to analyze!")
        
        with tab2:
            st.subheader("📰 Example Articles")
            
            examples = {
                "✅ Real News": """Washington - The President announced new economic policies during a press conference at the White House today. The comprehensive plan aims to address inflation concerns and includes measures to support small businesses across the country. Economic advisors present at the meeting outlined specific strategies for implementation over the next fiscal year.""",
                "❌ Fake News": """BREAKING NEWS: Scientists discover aliens living among us! Government officials have been hiding the truth for decades. Shocking evidence reveals they walk among us every day. Click here to learn the truth they don't want you to know! Share this before it gets deleted! This will change everything you believe!"""
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📰 Try Real News", use_container_width=True):
                    st.text_area("Example:", examples["✅ Real News"], height=200, key="real_ex")
            
            with col2:
                if st.button("🚨 Try Fake News", use_container_width=True):
                    st.text_area("Example:", examples["❌ Fake News"], height=200, key="fake_ex")
        
        with tab3:
            st.subheader("🧠 How It Works")
            
            st.markdown("""
            ### 1️⃣ Data Preprocessing
            - Combines title and text
            - Removes URLs, numbers, special characters
            - Converts to lowercase
            - Removes stopwords
            
            ### 2️⃣ LSTM Model Architecture
            \`\`\`
            Input Text
                ↓
            Embedding Layer (10,000 words → 128 dimensions)
                ↓
            LSTM Layer (128 units with dropout)
                ↓
            Dense Layer (64 units, ReLU)
                ↓
            Dropout Layer (30%)
                ↓
            Output Layer (Sigmoid → 0 or 1)
            \`\`\`
            
            ### 3️⃣ Prediction
            - Text is preprocessed
            - Converted to sequences
            - Fed through LSTM
            - Output: 0-1 probability
            - > 0.5 = Real, ≤ 0.5 = Fake
            
            ### 📊 Model Performance
            - Training Accuracy: ~92-95%
            - Test Accuracy: ~88-92%
            - Dataset: 20,000+ articles
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with TensorFlow, Keras & Streamlit | LSTM Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()`
