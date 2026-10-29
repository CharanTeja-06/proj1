"""
Real-time ISL Alphabet Recognition via Webcam

Key Features & Improvements:
1. MediaPipe Hand Detection:
   - Uses MediaPipe Tasks HandLandmarker (compatible with MediaPipe 1.x+)
   - Automatic fallback to MediaPipe Solutions (0.x) or OpenCV skin-contour
   - When no hand is visible, prediction is disabled (displays 'Hand: NOT DETECTED')
2. Preprocessing & Class Alignment:
   - Uses identical preprocess_input() and CLASSES list directly from data.py
3. Dynamic Hand ROI:
   - Crops hand bounding box with margin and resizes to (224, 224) matching model training input
4. Robust Smoothing & Confidence Thresholding:
   - Configurable CONFIDENCE_THRESHOLD (default 0.55) rejects uncertain predictions
   - Temporal deque smoothing over last 10 frames eliminates single-frame jitter
5. Hold-to-Confirm Commit:
   - Stable gesture must be held for 15 frames before appending to text buffer
   - Visual progress bar indicates confirmation progress
6. Top-3 Prediction Display:
   - Displays top 3 letter probabilities to easily monitor visually similar letters (e.g. S vs Z)
7. Text Composition Controls:
   - [Q] Quit | [C] Clear buffer | [B] Backspace | [SPACE] Add space
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque, Counter

# Authoritative preprocessing, classes, and input size
from data import preprocess_input, CLASSES, IMG_SIZE

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"
HAND_TASK_PATH = ROOT / "models" / "hand_landmarker.task"

# ─── Tunable Parameters ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55   # Minimum confidence to accept a prediction
HISTORY_LEN          = 10     # Number of frames in temporal smoothing window
STABLE_FRAMES        = 15     # Consecutive frames required to commit a letter
ROI_MARGIN           = 30     # Pixel margin around hand bounding box

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm base
]


class HandDetector:
    """
    Robust multi-backend hand detector:
    1. MediaPipe Tasks HandLandmarker (MediaPipe 1.x)
    2. MediaPipe Solutions Hands (MediaPipe 0.x fallback)
    3. OpenCV HSV skin-contour fallback
    """
    def __init__(self, task_path=HAND_TASK_PATH):
        self.backend = None
        self.detector = None
        self.mp = None

        # Try MediaPipe Tasks API
        if task_path.exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                options = vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.55,
                    min_hand_presence_confidence=0.55,
                    min_tracking_confidence=0.50
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.backend = "tasks"
                self.mp = mp
                print(f"[HandDetector] Using MediaPipe Tasks Landmarker ({task_path.name})")
            except Exception as e:
                print(f"[HandDetector] MediaPipe Tasks init failed: {e}")

        # Try legacy MediaPipe Solutions API
        if self.backend is None:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    self.detector = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.55,
                        min_tracking_confidence=0.50
                    )
                    self.backend = "solutions"
                    self.mp = mp
                    print("[HandDetector] Using MediaPipe Solutions Hands")
            except Exception as e:
                print(f"[HandDetector] MediaPipe Solutions init failed: {e}")

        # Fallback to HSV skin detector
        if self.backend is None:
            self.backend = "contour"
            print("[HandDetector] Using OpenCV Skin-Contour fallback detector")

    def detect(self, frame_bgr):
        """
        Detects hand in BGR frame.
        Returns: (detected: bool, bbox: (x1, y1, x2, y2) or None, landmarks_pts: [(x, y), ...] or None)
        """
        h, w = frame_bgr.shape[:2]

        if self.backend == "tasks":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(mp_img)
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                lms = res.hand_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lms]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = max(0, min(xs) - ROI_MARGIN)
                y1 = max(0, min(ys) - ROI_MARGIN)
                x2 = min(w, max(xs) + ROI_MARGIN)
                y2 = min(h, max(ys) + ROI_MARGIN)
                if x2 > x1 + 20 and y2 > y1 + 20:
                    return True, (x1, y1, x2, y2), pts

        elif self.backend == "solutions":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.detector.process(rgb)
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lms.landmark]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = max(0, min(xs) - ROI_MARGIN)
                y1 = max(0, min(ys) - ROI_MARGIN)
                x2 = min(w, max(xs) + ROI_MARGIN)
                y2 = min(h, max(ys) + ROI_MARGIN)
                if x2 > x1 + 20 and y2 > y1 + 20:
                    return True, (x1, y1, x2, y2), pts

        elif self.backend == "contour":
            # HSV skin color detection
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70], dtype=np.uint8)
            upper = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    x, y, cw, ch = cv2.boundingRect(c)
                    x1 = max(0, x - ROI_MARGIN)
                    y1 = max(0, y - ROI_MARGIN)
                    x2 = min(w, x + cw + ROI_MARGIN)
                    y2 = min(h, y + ch + ROI_MARGIN)
                    return True, (x1, y1, x2, y2), None

        return False, None, None

    def close(self):
        if self.backend == "solutions" and hasattr(self.detector, "close"):
            self.detector.close()


def draw_skeleton(frame, pts):
    """Draws hand skeleton connections and landmark points."""
    if not pts:
        return
    for start, end in HAND_CONNECTIONS:
        if start < len(pts) and end < len(pts):
            cv2.line(frame, pts[start], pts[end], (0, 255, 120), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def get_confidence_color(conf):
    """Green for high (>=75%), Orange for moderate (>=55%), Red otherwise."""
    if conf >= 0.75:
        return (0, 220, 0)
    elif conf >= CONFIDENCE_THRESHOLD:
        return (0, 165, 255)
    return (0, 0, 220)


def run_webcam():
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available. Please verify camera connection/permissions.")

    history = deque(maxlen=HISTORY_LEN)
    candidate = None
    candidate_frames = 0
    last_added = None
    text_buffer = ""

    img_h, img_w = IMG_SIZE

    print("\nStarting webcam feed. Press [Q] to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # Mirror frame for natural interaction
            fh, fw = frame.shape[:2]

            hand_detected, bbox, landmarks = detector.detect(frame)

            if hand_detected and bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]

                # Convert BGR crop to RGB and resize
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(crop_rgb, (img_w, img_h)).astype(np.float32)

                # Preprocess identically to training
                tensor = preprocess_input(resized)
                batch = np.expand_dims(tensor, axis=0)

                probs = model.predict(batch, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])

                # Draw skeleton and hand bounding box
                if landmarks:
                    draw_skeleton(frame, landmarks)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            else:
                idx = -1
                confidence = 0.0
                probs = np.zeros(len(CLASSES), dtype=np.float32)

            # ─── Temporal Smoothing ──────────────────────────────────────────
            if hand_detected and confidence >= CONFIDENCE_THRESHOLD:
                history.append(idx)
            else:
                history.append(-1)

            valid_history = [x for x in history if x >= 0]
            if valid_history:
                stable_idx, _ = Counter(valid_history).most_common(1)[0]
                stable_letter = CLASSES[stable_idx]
            else:
                stable_idx = -1
                stable_letter = "---"

            # ─── Hold-to-Confirm Behavior ────────────────────────────────────
            if hand_detected and confidence >= CONFIDENCE_THRESHOLD and stable_idx >= 0:
                if stable_idx == candidate:
                    candidate_frames += 1
                else:
                    candidate = stable_idx
                    candidate_frames = 1

                if candidate_frames >= STABLE_FRAMES and candidate != last_added:
                    text_buffer += CLASSES[candidate]
                    last_added = candidate
            else:
                candidate = None
                candidate_frames = 0
                if not hand_detected:
                    last_added = None  # Reset so the same letter can be re-signed after lifting hand

            # ─── HUD Display ─────────────────────────────────────────────────
            # Top dark header banner
            cv2.rectangle(frame, (0, 0), (fw, 125), (25, 25, 25), -1)

            # Hand status indicator
            if hand_detected:
                hand_status = "Hand: DETECTED"
                hand_status_color = (0, 220, 0)
            else:
                hand_status = "Hand: NOT DETECTED"
                hand_status_color = (0, 0, 220)
            cv2.putText(frame, hand_status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_status_color, 2)

            # Prediction & confidence
            pred_color = get_confidence_color(confidence)
            if hand_detected and confidence >= CONFIDENCE_THRESHOLD:
                pred_str = f"Prediction: {stable_letter} ({confidence:.0%})"
            elif hand_detected:
                pred_str = f"Prediction: Uncertain ({confidence:.0%})"
            else:
                pred_str = "Prediction: ---"
            cv2.putText(frame, pred_str, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2)

            # Top-3 predictions
            if hand_detected and probs.sum() > 0:
                top3_indices = np.argsort(probs)[-3:][::-1]
                top3_str = "  ".join(f"{CLASSES[i]}: {probs[i]:.0%}" for i in top3_indices)
                cv2.putText(frame, f"Top-3: {top3_str}", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 1)

            # Progress bar for hold-to-confirm
            if candidate_frames > 0:
                bar_max_w = int(fw * 0.5)
                bar_w = int(bar_max_w * min(1.0, candidate_frames / STABLE_FRAMES))
                bar_x = 15
                bar_y = 98
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 14), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (0, 220, 0), -1)
                cv2.putText(frame, f"Holding {CLASSES[candidate]} ({candidate_frames}/{STABLE_FRAMES})",
                            (bar_x + bar_max_w + 12, bar_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Bottom dark banner for text buffer
            cv2.rectangle(frame, (0, fh - 65), (fw, fh), (25, 25, 25), -1)
            display_text = text_buffer[-35:] if len(text_buffer) > 35 else text_buffer
            cv2.putText(frame, f"Text: {display_text}", (15, fh - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "[Q] Quit  [C] Clear  [B] Backspace  [SPACE] Space",
                        (15, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("ISL Alphabet Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("c"):
                text_buffer = ""
                last_added = None
            elif key == ord("b"):
                text_buffer = text_buffer[:-1]
                last_added = None
            elif key == ord(" "):
                text_buffer += " "
                last_added = None

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
