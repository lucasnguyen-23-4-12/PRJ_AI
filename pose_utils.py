import math
import numpy as np
import time
import cv2
import mediapipe as mp
import threading
from playsound import playsound

# --------------------------- UTILITIES ---------------------------
def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def vector_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

# ----------------------- HEAD DOWN ANGLE -----------------------
_down_state = {"downing": False, "start_time": None, "total_time": 0.0}
def head_down_angle(keypoints, down_threshold=30, is_turning=False):
    """
    Tính góc cúi đầu & đếm thời gian cúi đầu.
    - Khi góc cúi đầu > down_threshold → bắt đầu đếm.
    - Khi ngẩng lên hoặc đang quay đầu → dừng đếm.
    """
    global _down_state

    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    nose = keypoints[0]
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    # ✅ Sửa ở đây: tính góc giữa vector (nose - mid_shoulder) và vector dọc (0, -1)
    angle = vector_angle(nose - mid_shoulder, np.array([0, -1]))
    down_now = angle > down_threshold

    # ❌ Nếu đang quay đầu thì dừng đếm cúi đầu
    if is_turning:
        if _down_state["downing"]:
            _down_state["total_time"] += time.time() - _down_state["start_time"]
            _down_state.update({"downing": False, "start_time": None})
        return angle, False, round(_down_state["total_time"], 2)

    # ✅ Bình thường: đếm thời gian cúi đầu
    if down_now and not _down_state["downing"]:
        _down_state.update({"downing": True, "start_time": time.time()})
    elif down_now:
        _down_state["total_time"] += time.time() - _down_state["start_time"]
        _down_state["start_time"] = time.time()
    elif not down_now and _down_state["downing"]:
        _down_state["total_time"] += time.time() - _down_state["start_time"]
        _down_state.update({"downing": False, "start_time": None})

    return angle, down_now, round(_down_state["total_time"], 2)


# ----------------------- HEAD TURN ANGLE -----------------------
_turn_state = {"turning": False, "start_time": None, "total_time": 0.0, "direction": None}
def head_turn_angle(keypoints, turn_threshold=35):
    global _turn_state
    nose, left_ear, right_ear = keypoints[0], keypoints[3], keypoints[4]
    ear_center_x = (left_ear[0] + right_ear[0]) / 2
    head_width = distance(left_ear, right_ear)
    if head_width == 0:
        return 0, None, _turn_state["total_time"]
    horizontal_diff = nose[0] - ear_center_x
    ratio = abs(horizontal_diff) / head_width
    angle = ratio * 90
    direction = "right" if horizontal_diff > 0 else "left"
    turning_now = angle > turn_threshold

    if turning_now and not _turn_state["turning"]:
        _turn_state.update({"turning": True, "start_time": time.time(), "direction": direction})
    elif turning_now:
        _turn_state["total_time"] += time.time() - _turn_state["start_time"]
        _turn_state["start_time"] = time.time()
    elif not turning_now and _turn_state["turning"]:
        _turn_state["total_time"] += time.time() - _turn_state["start_time"]
        _turn_state.update({"turning": False, "direction": None, "start_time": None})
    return angle, _turn_state["direction"], round(_turn_state["total_time"], 2)


# ----------------------- DROWSINESS DETECTION -----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.17
FRAME_SLEEP = 80
_closed_frames = 0

def _play_alarm():
    try:
        playsound("alert.wav")
    except:
        print("⚠️ Không phát được âm thanh cảnh báo")

def _ear_ratio(landmarks, eye_points):
    eye = np.array([(landmarks[p].x, landmarks[p].y) for p in eye_points])
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h)

def detect_drowsiness(frame):
    global _closed_frames
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    sleepy, ear_value = False, 0.0
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_ear = _ear_ratio(landmarks.landmark, LEFT_EYE)
            right_ear = _ear_ratio(landmarks.landmark, RIGHT_EYE)
            ear_value = (left_ear + right_ear) / 2.0
            if ear_value < EAR_THRESH:
                _closed_frames += 1
            else:
                _closed_frames = 0
            if _closed_frames >= FRAME_SLEEP:
                sleepy = True
                threading.Thread(target=_play_alarm).start()
    return sleepy, round(ear_value, 3)
