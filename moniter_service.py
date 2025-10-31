import os
import io
import threading
import time
import numpy as np
import cv2
import pyaudio
import soundfile as sf
import librosa
from flask import Flask, Response, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from playsound import playsound

# ------------------ Flask Setup ------------------
app = Flask(__name__)
CORS(app)

# ------------------ Global State ------------------
latest_status = {
    "crying_detected": False,
    "reason": "No Cry",
    "temperature": "25.0 C"
}

# ------------------ Load Model ------------------
MODEL_PATH = "baby_cry_classifier.h5"
model = load_model(MODEL_PATH)

LABELS = ["Hungry", "Tired", "Belly Pain", "Burping",
          "Discomfort", "Sad", "Laughing", "No Cry"]

# ------------------ Audio Monitoring ------------------
CHUNK = 22050
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RHYME_PATH = "rhymes.wav"

def preprocess_audio_in_memory(audio_bytes, sr_target=22050):
    """Convert BytesIO audio to (1,40) MFCC features."""
    try:
        audio_bytes.seek(0)
        data, sr = sf.read(audio_bytes, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != sr_target:
            data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
        mfcc = librosa.feature.mfcc(y=data, sr=sr_target, n_mfcc=40)
        return np.mean(mfcc, axis=1).reshape(1, -1)
    except Exception as e:
        print(f"❌ Audio preprocessing error: {e}")
        return None

def play_rhyme():
    if os.path.exists(RHYME_PATH):
        try:
            playsound(RHYME_PATH)
        except Exception as e:
            print(f"❌ Rhyme playback error: {e}")

def audio_monitoring_loop():
    global latest_status
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return

    print("🎤 Audio monitoring started...")
    last_pred_time = 0  # control prediction frequency

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if np.max(np.abs(audio_np)) > 0:
                audio_np /= np.max(np.abs(audio_np))

            # Only predict every 3 seconds
            if time.time() - last_pred_time > 3:
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, audio_np, RATE, format='WAV')
                features = preprocess_audio_in_memory(audio_bytes)
                if features is None:
                    continue

                prediction = model.predict(features, verbose=0)
                label_index = np.argmax(prediction, axis=1)[0]
                reason = LABELS[label_index]

                latest_status["crying_detected"] = reason != "No Cry"
                latest_status["reason"] = reason

                if latest_status["crying_detected"]:
                    threading.Thread(target=play_rhyme, daemon=True).start()

                last_pred_time = time.time()

            time.sleep(0.1)  # reduce CPU usage

        except Exception as e:
            print(f"❌ Audio loop error: {e}")
            time.sleep(1)

# ------------------ Video Streaming ------------------
camera_lock = threading.Lock()
camera = None

def get_camera():
    for idx in range(3):
        cam = cv2.VideoCapture(idx)
        # Lower resolution → less CPU usage
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        time.sleep(1)
        ret, frame = cam.read()
        if ret and np.mean(frame) > 10:
            return cam
        cam.release()
    return None

def ensure_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = get_camera()

def generate_frames():
    global camera
    ensure_camera()
    while camera is not None and camera.isOpened():
        ret, frame = camera.read()
        if not ret or np.mean(frame) < 10:
            ensure_camera()
            continue

        cv2.putText(frame, f"Status: {'Crying' if latest_status['crying_detected'] else 'Monitoring'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if latest_status['crying_detected'] else (0, 255, 0), 2)
        cv2.putText(frame, f"Reason: {latest_status['reason']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ------------------ Flask Routes ------------------
@app.route('/')
def home():
    return """
    <h1>👶 Baby Monitoring Server</h1>
    <p><a href='/status'>/status</a> → JSON status</p>
    <p><a href='/video_feed'>/video_feed</a> → Live camera</p>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(latest_status)

# ------------------ Start Audio Thread ------------------
audio_thread = threading.Thread(target=audio_monitoring_loop, daemon=True)
audio_thread.start()

if __name__ == "__main__":
    print("🚀 Baby monitoring server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
