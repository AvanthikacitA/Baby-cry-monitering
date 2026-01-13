import io
import time
import threading
import numpy as np
import cv2
import pyaudio
import librosa
import serial
from collections import deque
from flask import Flask, Response, jsonify, render_template_string
from tensorflow.keras.models import load_model

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)

# ---------------- GLOBAL STATE ----------------
latest_status = {
    "crying_detected": False,
    "reason": "No Cry",
    "confidence": 0.0,
    "temperature": "--"
}

# ---------------- LOAD AI MODEL ----------------
MODEL_PATH = "baby_cry_classifier.h5"
model = load_model(MODEL_PATH)

LABELS = [
    "Hungry", "Tired", "Belly Pain", "Burping",
    "Discomfort", "Sad", "Laughing", "No Cry"
]

# ---------------- AUDIO PARAMETERS ----------------
RATE = 22050
CHUNK = RATE
CHANNELS = 1
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 0.02
prob_history = deque(maxlen=5)

# ---------------- SERIAL (ESP32) ----------------
SERIAL_PORT = "COM3"   # 🔴 CHANGE THIS
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(audio_np):
    mfcc = librosa.feature.mfcc(y=audio_np, sr=RATE, n_mfcc=40)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# ---------------- ESP32 TEMPERATURE THREAD ----------------
def temperature_loop():
    global latest_status
    while True:
        try:
            if ser.in_waiting:
                temp = ser.readline().decode().strip()
                latest_status["temperature"] = f"{float(temp):.1f} °C"
        except:
            pass
        time.sleep(2)

# ---------------- AUDIO MONITORING ----------------
def audio_monitoring_loop():
    global latest_status
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_np**2))

        if rms < SILENCE_THRESHOLD:
            latest_status["crying_detected"] = False
            latest_status["reason"] = "No Cry"
            latest_status["confidence"] = 0.0
            prob_history.clear()
            continue

        audio_np /= np.max(np.abs(audio_np)) + 1e-6
        features = extract_features(audio_np)
        prediction = model.predict(features, verbose=0)
        prob_history.append(prediction[0])

        avg_probs = np.mean(prob_history, axis=0)
        idx = np.argmax(avg_probs)
        conf = np.max(avg_probs)

        if conf > 0.7 and LABELS[idx] != "No Cry":
            latest_status["crying_detected"] = True
            latest_status["reason"] = LABELS[idx]
            latest_status["confidence"] = float(conf)
        else:
            latest_status["crying_detected"] = False
            latest_status["reason"] = "No Cry"
            latest_status["confidence"] = float(conf)

# ---------------- VIDEO STREAM ----------------
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        cv2.putText(frame, f"Status: {'Crying' if latest_status['crying_detected'] else 'Monitoring'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if latest_status['crying_detected'] else (0, 255, 0), 2)

        cv2.putText(frame, f"Reason: {latest_status['reason']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Temperature: {latest_status['temperature']}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def home():
    return render_template_string("""
    <h1>👶 Baby Monitoring Dashboard</h1>
    <img src="/video_feed" width="640"><br><br>
    <h3>Status: <span id="status"></span></h3>
    <h3>Reason: <span id="reason"></span></h3>
    <h3>Confidence: <span id="confidence"></span></h3>
    <h3>Temperature: <span id="temperature"></span></h3>

    <script>
    async function update() {
        let r = await fetch('/status');
        let d = await r.json();
        status.innerText = d.crying_detected ? "Crying" : "Monitoring";
        reason.innerText = d.reason;
        confidence.innerText = d.confidence.toFixed(2);
        temperature.innerText = d.temperature;
    }
    setInterval(update, 2000);
    update();
    </script>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(latest_status)

# ---------------- START THREADS ----------------
threading.Thread(target=audio_monitoring_loop, daemon=True).start()
threading.Thread(target=temperature_loop, daemon=True).start()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚀 Baby Monitoring System Running...")
    app.run(port=5000, debug=False)
