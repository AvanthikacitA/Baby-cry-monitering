import os
import librosa
import numpy as np

# Define the paths to your data folders.
DATA_PATH = "C:\\Users\\Avanthika A\\Desktop\\baby_cry_detector"
CLASS_LABELS = {
    # Must match your dataset folder names
    "hungry": 0,
    "lonely": 1,
    "burping": 2,
    "belly pain": 3,
    "cold_hot": 4,
    "discomfort": 5,
    "laugh": 6,
    "noise": 7,
    "scared": 8,
    "silence": 9,
    "tired": 10
}

def extract_features(file_path, num_mfcc=40, max_pad_len=174):
    """
    Extracts MFCC features from an audio file.
    Returns a 1D numpy array of length `num_mfcc`.
    """
    try:
        # librosa.load accepts path-like. If user passes file-like elsewhere, they will
        # use a temporary file approach (handled in other scripts).
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
        # pad or truncate
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        # return averaged feature (consistent with your earlier code)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_path=DATA_PATH, class_labels=CLASS_LABELS):
    """
    Loads audio files, extracts features, and returns (features, labels).
    """
    all_features = []
    all_labels = []
    supported_formats = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    for class_name, label in class_labels.items():
        class_folder = os.path.join(data_path, class_name)
        if not os.path.exists(class_folder):
            print(f"Folder not found: {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            if filename.lower().endswith(supported_formats):
                file_path = os.path.join(class_folder, filename)
                features = extract_features(file_path)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label)

    if len(all_features) == 0:
        return np.array([]), np.array([])
    return np.array(all_features), np.array(all_labels)

if __name__ == "__main__":
    features, labels = load_data()
    if features.size == 0:
        print("No audio data found. Please check your folder structure and file formats.")
    else:
        print(f"Loaded {len(features)} samples.")
