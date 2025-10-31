# pyright: reportMissingImports=false
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from data_processor import load_data, DATA_PATH, CLASS_LABELS

# Load data
features, labels = load_data(DATA_PATH, CLASS_LABELS)
if features.size == 0:
    print("No audio data found. Cannot train the model.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

def create_multi_class_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_multi_class_model(input_shape=X_train.shape[1], num_classes=len(CLASS_LABELS))

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
print("Model training completed.")

model_file_path = os.path.join(DATA_PATH, 'baby_cry_classifier.h5')
model.save(model_file_path)
print(f"\nModel saved successfully to: {model_file_path}")