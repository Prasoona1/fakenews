import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import os
import sys
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 FAKE NEWS DETECTION - LSTM TRAINING")
print("="*60)

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Check for dataset
if not os.path.exists('train.csv'):
    print("❌ ERROR: train.csv not found!")
    print("📥 Please download from: https://www.kaggle.com/c/fake-news/data")
    sys.exit(1)

print("\\n📂 Loading dataset...")
df = pd.read_csv('train.csv')
print(f"✅ Loaded {len(df)} articles")

# Combine title and text
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("\\n🧹 Preprocessing text...")
df['cleaned_content'] = df['content'].apply(preprocess_text)

# Create visualizations directory
print("\\n📊 Creating visualizations...")
os.makedirs('visualizations', exist_ok=True)

# Word clouds
fake_text = ' '.join(df[df['label']==0]['cleaned_content'].head(1000))
real_text = ' '.join(df[df['label']==1]['cleaned_content'].head(1000))

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
wc = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.imshow(wc, interpolation='bilinear')
plt.title('Fake News - Common Words', fontsize=16, fontweight='bold')
plt.axis('off')

plt.subplot(1,2,2)
wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.imshow(wc, interpolation='bilinear')
plt.title('Real News - Common Words', fontsize=16, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('visualizations/wordclouds.png', dpi=150, bbox_inches='tight')
print("✅ Word clouds saved")

# Prepare data
X = df['cleaned_content'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\\n📊 Dataset split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Tokenization
print("\\n🔤 Tokenizing text...")
max_words = 10000
max_len = 300

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Build model
print("\\n🏗️  Building LSTM model...")
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\\n📋 Model Summary:")
model.summary()

# Train
print("\\n🚀 Training model...")
print("⏳ This will take 5-15 minutes...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/training_history.png', dpi=150, bbox_inches='tight')
print("\\n✅ Training history saved")

# Evaluate
print("\\n📊 Evaluating model...")
y_pred = (model.predict(X_test_pad, verbose=0) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"\\n{'='*60}")
print(f"🎯 FINAL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*60}")

print("\\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('visualizations/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\\n✅ Confusion matrix saved")

# Save model
print("\\n💾 Saving model and tokenizer...")
model.save('fake_news_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\\n" + "="*60)
print("✨ TRAINING COMPLETE!")
print("="*60)
print("📁 Files created:")
print("   - fake_news_model.h5 (trained model)")
print("   - tokenizer.pkl (text tokenizer)")
print("   - visualizations/ (charts and graphs)")
print("\\n🚀 Run: streamlit run app.py")
print("="*60)`,
