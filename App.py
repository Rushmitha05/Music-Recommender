import sys
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import webbrowser
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QComboBox, QLineEdit, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

# -------------------------
# Feedback file management
# -------------------------
FEEDBACK_FILE = "feedback.json"
try:
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)
    else:
        feedback_data = {}
except (json.JSONDecodeError, FileNotFoundError):
    feedback_data = {}

# -------------------------
# Model loading with checks
# -------------------------
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.npy"

print("🔍 Current working directory:", os.getcwd())
print("📂 Files here:", os.listdir("."))

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        from keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        labels = np.load(LABELS_PATH, allow_pickle=True)
        labels = np.array([label.capitalize() for label in labels])
        model_loaded = True
        print("✅ Model and labels loaded successfully.")
    else:
        raise FileNotFoundError("model.h5 or labels.npy not found.")
except Exception as e:
    print("⚠️ Model not loaded:", e)
    model = None
    labels = None
    model_loaded = False

# -------------------------
# Mediapipe setup
# -------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                enable_segmentation=False,
                                refine_face_landmarks=True)

# -------------------------
# Main App
# -------------------------
class EmotionMusicApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 Emotion-Based Music Recommender")
        self.setGeometry(100, 100, 900, 650)
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #f0f0f0; font-family: 'Segoe UI'; font-size: 15px; }
            QPushButton { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #4e54c8, stop:1 #8f94fb); color: white; border-radius: 10px; padding: 8px; }
            QPushButton:hover { background-color: #6C63FF; }
            QLineEdit, QComboBox { background-color: #2b2b2b; color: white; border: 1px solid #555; border-radius: 8px; padding: 6px; }
            QLabel { font-weight: bold; }
        """)

        # Layouts
        self.layout = QVBoxLayout()
        self.title_label = QLabel("🎧 Emotion-Based Music Recommender")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.layout.addWidget(self.title_label)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #555; border-radius: 10px;")
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Input controls
        controls_layout = QHBoxLayout()
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["Camera", "Text"])
        self.input_mode_combo.currentTextChanged.connect(self.input_mode_changed)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter Emotion (e.g. Happy)")
        self.text_input.setVisible(False)
        controls_layout.addWidget(self.input_mode_combo)
        controls_layout.addWidget(self.text_input)
        self.layout.addLayout(controls_layout)

        # Emotion label
        self.emotion_label = QLabel("🎭 Detected Emotion: Not Detected")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.emotion_label)

        # Language input
        self.lang_input = QLineEdit()
        self.lang_input.setPlaceholderText("Enter Language (e.g. Hindi, English, Tamil)")
        self.layout.addWidget(self.lang_input)

        # Platform selection
        self.platform_dropdown = QComboBox()
        self.platform_dropdown.addItems(["YouTube", "Spotify", "Apple Music"])
        self.layout.addWidget(self.platform_dropdown)

        # Buttons
        btn_layout = QHBoxLayout()
        self.detect_button = QPushButton("🎥 Start Detection")
        self.detect_button.clicked.connect(self.start_detection)
        btn_layout.addWidget(self.detect_button)

        self.music_button = QPushButton("🎶 Recommend Music")
        self.music_button.clicked.connect(self.recommend_music)
        btn_layout.addWidget(self.music_button)

        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

        # Video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.detected_emotion = "Not Detected"
        self.feedback_given = False

    def input_mode_changed(self, mode):
        if mode == "Text":
            self.text_input.setVisible(True)
            self.stop_camera()
            self.emotion_label.setText("🎭 Detected Emotion: Not Detected")
        else:
            self.text_input.setVisible(False)
            self.emotion_label.setText("🎭 Detected Emotion: Not Detected")

    def start_detection(self):
        self.stop_camera()  # make sure old camera is released
        self.feedback_given = False
        if self.input_mode_combo.currentText() == "Camera":
            if not model_loaded:
                QMessageBox.warning(self, "Model Error", "Model not loaded! Please provide model.h5 and labels.npy.")
                return
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Webcam not found!")
                return
            self.timer.start(30)
        else:
            self.detected_emotion = self.text_input.text().strip().capitalize()
            if not self.detected_emotion:
                QMessageBox.warning(self, "Input Error", "Please enter a valid emotion.")
                return
            self.emotion_label.setText(f"🎭 Detected Emotion: {self.detected_emotion}")

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        if not self.feedback_given and self.detected_emotion != "Not Detected":
            self.collect_feedback()
            self.feedback_given = True

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        features = []
        if results.face_landmarks:
            origin = results.face_landmarks.landmark[1]
            for lm in results.face_landmarks.landmark:
                features.extend([lm.x - origin.x, lm.y - origin.y])
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y])
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y])

        features = features[:1020] + [0.0] * max(0, 1020 - len(features))
        features_np = np.array(features).reshape(1, -1)

        if model_loaded and results.face_landmarks:
            pred = model.predict(features_np, verbose=0)
            label = labels[np.argmax(pred)]
            self.detected_emotion = label
            self.emotion_label.setText(f"🎭 Detected Emotion: {label}")
        else:
            self.emotion_label.setText("😶 No face detected")

        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def recommend_music(self):
        emotion = self.detected_emotion.lower()
        lang = self.lang_input.text().strip().lower()

        if not emotion or emotion == "not detected":
            QMessageBox.warning(self, "No Emotion", "Please detect an emotion first.")
            return

        platform = self.platform_dropdown.currentText()
        if not lang:
            lang = "english"  # default

        if emotion == "happy":
            query = f"{lang} pop dance edm songs"
        elif emotion == "sad":
            query = f"{lang} acoustic soft rock songs"
        elif emotion in ["angry", "fearful", "disgusted"]:
            query = f"{lang} calm relaxing positive songs"
        elif emotion in ["neutral", "surprised"]:
            query = f"{lang} party dance songs"
        else:
            query = f"{lang} {emotion} songs"

        query_encoded = query.replace(" ", "+")
        urls = {
            "YouTube": f"https://www.youtube.com/results?search_query={query_encoded}",
            "Spotify": f"https://open.spotify.com/search/{query_encoded}",
            "Apple Music": f"https://music.apple.com/us/search?term={query_encoded}"
        }

        webbrowser.open(urls.get(platform, urls["YouTube"]))
        self.emotion_label.setText(f"🎵 Searching for '{query}' on {platform}")

    def collect_feedback(self):
        res = QMessageBox.question(self, "Feedback",
                                   f"Was the detected emotion '{self.detected_emotion}' accurate?",
                                   QMessageBox.Yes | QMessageBox.No)
        emo = self.detected_emotion
        if emo not in feedback_data:
            feedback_data[emo] = {"Yes": 0, "No": 0}
        if res == QMessageBox.Yes:
            feedback_data[emo]["Yes"] += 1
        else:
            feedback_data[emo]["No"] += 1
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_data, f, indent=2)
        QMessageBox.information(self, "Thank You!", "Your feedback helps us improve the recommendations!")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionMusicApp()
    window.show()
    sys.exit(app.exec_())
