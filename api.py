import os
import io
import threading
import time
import numpy as np
import cv2
import pyaudio
import soundfile as sf
import librosa
from collections import deque
from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.models import load_model
import winsound  # Windows sound playback

# ------------------ Flask Setup ------------------
app = Flask(__name__)
CORS(app)

# ------------------ Global State ------------------
latest_status = {
    "crying_detected": False,
    "reason": "No Cry",
    "confidence": 0.0,
    "temperature": "25.0 C"
}

# ------------------ Load Model ------------------
MODEL_PATH = "baby_cry_classifier.h5"  # Ensure this file exists
model = load_model(MODEL_PATH)
LABELS = ["Hungry", "Tired", "Belly Pain", "Burping",
          "Discomfort", "Sad", "Laughing", "No Cry"]

# ------------------ Audio Parameters ------------------
CHUNK = 22050  # 1 second chunk at 22050Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
SILENCE_THRESHOLD = 0.02

prob_history = deque(maxlen=5)

# ------------------ Audio Feature Extraction ------------------
def extract_features(audio_np, sr=22050):
    try:
        mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

# ------------------ Sound Playback ------------------
def play_rhyme_once():
    RHYME_PATH = "rhymes.wav"
    if not os.path.exists(RHYME_PATH):
        print(f"❌ Rhyme file not found at {RHYME_PATH}")
        return
    winsound.PlaySound(RHYME_PATH, winsound.SND_FILENAME)

def play_monitoring_beep():
    try:
        winsound.Beep(800, 100)
    except Exception as e:
        print(f"❌ Monitoring beep error: {e}")

# ------------------ Audio Monitoring Loop ------------------
def audio_monitoring_loop():
    global latest_status, prob_history

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return

    rhyme_thread = None
    last_played_crying = False

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_np**2))

            if rms < SILENCE_THRESHOLD:
                latest_status["crying_detected"] = False
                latest_status["reason"] = "No Cry"
                latest_status["confidence"] = 0.0
                prob_history.clear()
                last_played_crying = False
                play_monitoring_beep()
                continue

            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np /= max_val + 1e-6
            else:
                continue

            features = extract_features(audio_np)
            if features is None:
                continue

            prediction = model.predict(features, verbose=0)
            prob_history.append(prediction[0])

            avg_probs = np.mean(prob_history, axis=0)
            max_prob = np.max(avg_probs)
            label_index = np.argmax(avg_probs)
            reason = LABELS[label_index]

            threshold = 0.7
            if max_prob > threshold and reason != "No Cry":
                latest_status["crying_detected"] = True
                latest_status["reason"] = reason
                latest_status["confidence"] = float(max_prob)
                if not last_played_crying:
                    if rhyme_thread is None or not rhyme_thread.is_alive():
                        rhyme_thread = threading.Thread(target=play_rhyme_once, daemon=True)
                        rhyme_thread.start()
                    last_played_crying = True
            else:
                latest_status["crying_detected"] = False
                latest_status["reason"] = "No Cry"
                latest_status["confidence"] = float(max_prob)
                last_played_crying = False
                play_monitoring_beep()

        except Exception as e:
            print(f"❌ Audio loop error: {e}")
            time.sleep(1)

# ------------------ Dummy Temperature Update ------------------
def dummy_temperature_loop():
    import random
    while True:
        temp = 24.0 + 3.0 * random.random()
        latest_status["temperature"] = f"{temp:.1f} C"
        time.sleep(5)  # update every 5 seconds

# ------------------ Video Streaming ------------------
camera_lock = threading.Lock()
camera = None

def get_camera():
    for idx in range(3):
        cam = cv2.VideoCapture(idx)
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
                    (0, 0, 255) if latest_status["crying_detected"] else (0, 255, 0), 2)
        cv2.putText(frame, f"Reason: {latest_status['reason']} (Conf: {latest_status['confidence']:.2f})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Temperature: {latest_status['temperature']}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ------------------ Flask Routes ------------------
@app.route('/')
def home():
    html = """
    <html>
    <head><title>👶 Baby Monitoring Dashboard</title></head>
    <body style="text-align:center; font-family:Arial;">
        <h1>Baby Monitoring Dashboard</h1>
        <h3>Status: <span id="status">Loading...</span></h3>
        <h3>Reason: <span id="reason">Loading...</span></h3>
        <h3>Confidence: <span id="confidence">0.0</span></h3>
        <h3>Temperature: <span id="temperature">--</span></h3>
        <img src="/video_feed" width="640" height="480"/>
        <script>
            async function fetchStatus() {
                try {
                    let res = await fetch('/status');
                    let data = await res.json();
                    document.getElementById("status").innerText = data.crying_detected ? "Crying" : "Monitoring";
                    document.getElementById("reason").innerText = data.reason;
                    document.getElementById("confidence").innerText = data.confidence.toFixed(2);
                    document.getElementById("temperature").innerText = data.temperature;
                } catch(e) { console.log("Status fetch error", e); }
            }
            setInterval(fetchStatus, 2000);
            fetchStatus();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(latest_status)

# ------------------ Start Threads ------------------
audio_thread = threading.Thread(target=audio_monitoring_loop, daemon=True)
audio_thread.start()

temp_thread = threading.Thread(target=dummy_temperature_loop, daemon=True)
temp_thread.start()

# ------------------ Main ------------------
if __name__ == "__main__":
    print("🚀 Baby monitoring server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
