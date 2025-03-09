################################          Training           ##############################################
import os
import librosa
import numpy as np
import pandas as pd
import joblib  # For saving label encoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Dataset Path (Update if necessary)
DATASET_PATH = "/content/tess_dataset/TESS Toronto emotional speech set data"

# Supported audio formats
SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True, delta=True):
    try:
          # Debugging progress
        X, sample_rate = librosa.load(file_path, sr=None)  # Auto-handles format conversion
        result = np.array([])

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)
            result = np.hstack((result, mfccs)) if result.size else mfccs

            if delta:
                delta_mfcc = librosa.feature.delta(mfccs)
                delta_mfcc = np.mean(delta_mfcc.T, axis=0)
                result = np.hstack((result, delta_mfcc))

        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma)) if result.size else chroma

        if mel:
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel_spectrogram)) if result.size else mel_spectrogram

        if result.size == 0:
            raise ValueError("Feature extraction failed")

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract file paths and labels
features, labels = [], []
emotion_dirs = os.listdir(DATASET_PATH)

for emotion in emotion_dirs:
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    if os.path.isdir(emotion_folder):
        for file_name in os.listdir(emotion_folder):
            if file_name.lower().endswith(SUPPORTED_FORMATS):  # Supports all formats
                file_path = os.path.join(emotion_folder, file_name)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels and save LabelEncoder
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
joblib.dump(encoder, "label_encoder.pkl")  # Save label encoder
print("LabelEncoder saved as 'label_encoder.pkl'.")

labels_categorical = to_categorical(labels_encoded)

# Feature Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_categorical, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build the LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(128, return_sequences=False),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model with callbacks
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32,
                    callbacks=[early_stopping, lr_reduction])

# Save the model
model.save("speech_emotion_recognition_model_optimized.h5")
print("Model saved as 'speech_emotion_recognition_model_optimized.h5'.")

# Plot training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Evaluate the model
eval_results = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_results[0]}\nTest Accuracy: {eval_results[1]}")






##############################         Testing           ##############################################


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import librosa
import joblib
import io
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model

# Force TensorFlow to use CPU (Fixes GPU errors)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#  Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


# Load the trained model and label encoder
MODEL_PATH = "speech_emotion_recognition_model_optimized (1).h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

try:
    model = load_model(MODEL_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    model_status = "Model loaded successfully"
except Exception as e:
    model_status = f"Model loading failed: {str(e)}"

# Feature extraction function
def extract_features(audio_data, sr):
    features = np.array([])

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    features = np.hstack((features, mfccs))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    features = np.hstack((features, chroma))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel = np.mean(mel.T, axis=0)
    features = np.hstack((features, mel))

    return features.reshape(1, -1, 1)

#  Health Check Route
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_status": model_status}

#  Speech Emotion Prediction Route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    audio_bytes = await file.read()

    # Convert bytes to an audio file for librosa
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Extract features
    features = extract_features(audio_data, sr)

    # Make prediction
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = encoder.inverse_transform([predicted_index])[0]

    return {"predicted_emotion": predicted_emotion}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)