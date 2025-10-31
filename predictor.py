import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('baby_cry_classifier.h5')

# Define the same feature extraction function as in data_processor.py
def extract_features(file_path, num_mfcc=40, max_pad_len=174):
    """
    Extracts MFCC features from an audio file.
    Args:
        file_path (str): The path to the audio file.
        num_mfcc (int): The number of MFCCs to extract.
        max_pad_len (int): The maximum length to pad or truncate the MFCC array.
    Returns:
        numpy.ndarray: The processed MFCC features.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
        
        # Pad or truncate the MFCC array to a fixed length for consistency
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# The CLASS_LABELS dictionary is copied here to ensure the labels match
CLASS_LABELS = {
    "hungry": 0, "lonely": 1, "burping": 2, "belly pain": 3,
    "cold_hot": 4, "discomfort": 5, "laugh": 6, "noise": 7,
    "scared": 8, "silence": 9, "tired": 10
}
# Create a reverse mapping for easy lookup of class names
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

def predict_cry(file_path):
    """
    Makes a prediction on a new audio file.
    Args:
        file_path (str): The path to the new audio file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Extract features from the new file
    features = extract_features(file_path)

    if features is not None:
        # The model expects a batch of samples, so reshape the single sample
        features = np.expand_dims(features, axis=0)

        # Make the prediction
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions)
        predicted_class = LABEL_TO_CLASS[predicted_label]
        confidence = predictions[0][predicted_label] * 100

        print(f"\nPrediction for {os.path.basename(file_path)}:")
        print(f"Predicted reason: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # Example usage: Replace 'path/to/your/new_audio.wav' with a real file path
    # You can test with one of the files from your dataset
    test_file_path = "C:\\Users\\Avanthika A\\Desktop\\baby_cry_detector\\hungry\\1f40790f-68f2-4e7e-845e-715bd97c82d0-1429979568422-1.7-m-04-hu.wav"
    
    predict_cry(test_file_path)
    
