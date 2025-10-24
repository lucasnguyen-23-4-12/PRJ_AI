import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime, timedelta
import supervision as sv
import pytesseract
from unidecode import unidecode
from pose_utils import head_down_angle, head_turn_angle, detect_drowsiness
from focus_logger import FocusLogger
from settings_loader import load_settings


# =====================================================
# 1️⃣ ĐỌC FILE CẤU HÌNH
# =====================================================
cfg = load_settings()

VIDEO_SOURCE = cfg["VIDEO_SOURCE"]
MODEL_PATH = cfg["MODEL_PATH"]
SESSION_DURATION = cfg["SESSION_DURATION"]
DOWN_THRESHOLD = cfg["DOWN_THRESHOLD"]
TURN_THRESHOLD = cfg["TURN_THRESHOLD"]
EAR_SLEEP_THRESHOLD = cfg["EAR_SLEEP_THRESHOLD"]
SLEEP_CONFIRM_TIME = cfg["SLEEP_CONFIRM_TIME"]
EAR_FRAMES_CONFIRM = cfg["EAR_FRAMES_CONFIRM"]
SPATIAL_TOLERANCE = cfg["SPATIAL_TOLERANCE"]
FRAME_LEARNING_PERIOD = cfg["FRAME_LEARNING_PERIOD"]
OUT_TIMEOUT = cfg["OUT_TIMEOUT"]
WINDOW_WIDTH = cfg["WINDOW_WIDTH"]
WINDOW_HEIGHT = cfg["WINDOW_HEIGHT"]
WARNING_BOX_SIZE = cfg["WARNING_BOX_SIZE"]
BLINK_SPEED = cfg["BLINK_SPEED"]
SUMMARY_FILE = cfg["SUMMARY_FILE"]
DETAIL_LOG_FILE = cfg["DETAIL_LOG_FILE"]
DISTRACTION_THRESHOLD = 0.50

# =====================================================
# 2️⃣ KHỞI TẠO
# =====================================================
model = YOLO(MODEL_PATH)
logger = FocusLogger()
cap = cv2.VideoCapture(VIDEO_SOURCE)
tracker = sv.ByteTrack()

start_time = datetime.now()
end_time = start_time + timedelta(seconds=SESSION_DURATION)
session_start_timestamp = time.time()

cv2.namedWindow("Meeting Focus Monitor (Spatial)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Meeting Focus Monitor (Spatial)", WINDOW_WIDTH, WINDOW_HEIGHT)

spatial_frames = {}
person_stats = {}
person_names = {}
meeting_log = []
frame_count = 0
learning_positions = []

# =====================================================
# 3️⃣ HÀM QUẢN LÝ SPATIAL FRAMES
# =====================================================
def find_or_create_frame_id(cx, cy):
    min_dist = float('inf')
    closest_frame_id = None
    for frame_id, frame_info in spatial_frames.items():
        fx, fy = frame_info["center"]
        dist = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
        if dist < SPATIAL_TOLERANCE and dist < min_dist:
            min_dist = dist
            closest_frame_id = frame_id

    if closest_frame_id is not None:
        return closest_frame_id

    new_frame_id = len(spatial_frames) + 1
    spatial_frames[new_frame_id] = {
        "center": (cx, cy),
        "last_seen": time.time(),
        "is_empty": False,
        "empty_start": None,
    }
    print(f"🆕 Created Frame {new_frame_id} at ({int(cx)}, {int(cy)})")
    return new_frame_id


def learn_spatial_layout(positions):
    if len(positions) < 10:
        return
    positions = np.array(positions)
    unique_positions = []
    for pos in positions:
        if all(np.linalg.norm(pos - upos) >= SPATIAL_TOLERANCE for upos in unique_positions):
            unique_positions.append(pos)
    spatial_frames.clear()
    for i, pos in enumerate(unique_positions):
        spatial_frames[i + 1] = {
            "center": (float(pos[0]), float(pos[1])),
            "last_seen": time.time(),
            "is_empty": False,
            "empty_start": None,
        }
        print(f"📍 Frame {i+1}: position ({int(pos[0])}, {int(pos[1])})")


def ensure_frame_stats(frame_id):
    if frame_id not in person_stats:
        person_stats[frame_id] = {
            "down_time": 0.0,
            "turn_time": 0.0,
            "ear": 0.0,
            "sleep_time": 0.0,
            "sleep_start": None,
            "is_sleeping": False,
            "state": "FOCUSED",
            "ear_buffer": [],
            "out_time": 0.0,
        }

# =====================================================
# 4️⃣ PHÁT HIỆN BUỒN NGỦ & TRÍCH XUẤT TÊN
# =====================================================
def detect_drowsiness_frame(face_crop, frame_id):
    if face_crop is None or face_crop.size == 0:
        return False, 0.4
    try:
        face_resized = cv2.resize(face_crop, (200, 200))
        sleepy, ear_value = detect_drowsiness(face_resized)
    except Exception:
        return False, 0.4

    if ear_value < 0.05 or ear_value > 0.45:
        return False, 0.3

    buf = person_stats.setdefault(frame_id, {}).setdefault("ear_buffer", [])
    buf.append(ear_value)
    if len(buf) > EAR_FRAMES_CONFIRM:
        buf.pop(0)
    avg_ear = float(np.mean(buf))
    sleepy_now = avg_ear < EAR_SLEEP_THRESHOLD

    # 🧠 Lưu trạng thái buồn ngủ theo thời gian thực
    stat = person_stats[frame_id]
    if "sleep_start" not in stat:
        stat["sleep_start"] = None

    # --- Nếu EAR thấp liên tục, coi như đang buồn ngủ ---
    if sleepy_now:
        # Bắt đầu đếm nếu chưa có sleep_start
        if stat["sleep_start"] is None:
            stat["sleep_start"] = time.time()

        # Nếu đã nhắm mắt liên tục đủ SLEEP_CONFIRM_TIME → chuyển sang trạng thái ngủ
        elif not stat.get("is_sleeping", False) and (time.time() - stat["sleep_start"]) >= SLEEP_CONFIRM_TIME:
            stat["is_sleeping"] = True
            print(f"😴 Frame {frame_id} bắt đầu ngủ (EAR={avg_ear:.2f})")

    # --- Nếu mở mắt lại ---
    else:
        if stat.get("is_sleeping", False):
            sleep_duration = time.time() - stat["sleep_start"]
            stat["sleep_time"] += sleep_duration
            print(f"✅ Frame {frame_id} tỉnh lại (ngủ {sleep_duration:.1f}s)")
        # Reset lại
        stat["is_sleeping"] = False
        stat["sleep_start"] = None

    return stat.get("is_sleeping", False), avg_ear


def extract_name_from_region(frame, x1, y1, x2, y2):
    try:
        name_region = frame[y2:min(y2 + 50, frame.shape[0]), x1:x2]
        if name_region.size == 0:
            return None
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, lang='vie', config='--psm 7').strip()
        return unidecode(text) if text else None
    except:
        return None

# =====================================================
# 5️⃣ VÒNG LẶP CHÍNH
# =====================================================
layout_learned = False
prev_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_time = time.time()
    delta_time = current_frame_time - prev_frame_time
    prev_frame_time = current_frame_time
    frame_count += 1
    results = model.predict(frame, verbose=False)
    now = time.time()
    frame_height, frame_width = frame.shape[:2]

    # Giai đoạn học layout
    if not layout_learned and frame_count <= FRAME_LEARNING_PERIOD:
        if results and hasattr(results[0], "boxes") and len(results[0].boxes.xyxy) > 0:
            for xyxy in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                learning_positions.append([cx, cy])
        cv2.putText(frame, f"Learning layout... {frame_count}/{FRAME_LEARNING_PERIOD}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if frame_count == FRAME_LEARNING_PERIOD:
            learn_spatial_layout(learning_positions)
            layout_learned = True
            print(f"\n✅ Layout learned! Total frames: {len(spatial_frames)}")
        cv2.imshow("Meeting Focus Monitor (Spatial)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    occupied_frames = set()
    if results and hasattr(results[0], "boxes") and len(results[0].boxes.xyxy) > 0:
        detections = sv.Detections.from_ultralytics(results[0])
        tracked = tracker.update_with_detections(detections)
        for xyxy, track_id, conf, cls in zip(tracked.xyxy, tracked.tracker_id,
                                             tracked.confidence, tracked.class_id):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            frame_id = find_or_create_frame_id(cx, cy)
            occupied_frames.add(frame_id)
            ensure_frame_stats(frame_id)

            spatial_frames[frame_id]["last_seen"] = now
            if spatial_frames[frame_id]["is_empty"]:
                if spatial_frames[frame_id]["empty_start"]:
                    person_stats[frame_id]["out_time"] += now - spatial_frames[frame_id]["empty_start"]
                spatial_frames[frame_id]["is_empty"] = False
                spatial_frames[frame_id]["empty_start"] = None

            if frame_id not in person_names or (frame_count % 30 == 0):
                detected_name = extract_name_from_region(frame, x1, y1, x2, y2)
                if detected_name:
                    person_names[frame_id] = detected_name
            display_name = person_names.get(frame_id, f"Frame {frame_id}")

            head_crop = frame[y1:y2, x1:x2]
            sleepy, ear_value = detect_drowsiness_frame(head_crop, frame_id)
            person_stats[frame_id]["ear"] = ear_value

            if sleepy:
                if not person_stats[frame_id]["is_sleeping"]:
                    person_stats[frame_id]["is_sleeping"] = True
                    person_stats[frame_id]["sleep_start"] = now
                    print(f"😴 {display_name} bắt đầu buồn ngủ")
            else:
                if person_stats[frame_id]["is_sleeping"]:
                    sleep_duration = now - person_stats[frame_id]["sleep_start"]
                    person_stats[frame_id]["sleep_time"] += sleep_duration
                    person_stats[frame_id]["is_sleeping"] = False
                    print(f"✅ {display_name} tỉnh lại (ngủ {sleep_duration:.1f}s)")

            direction = None
            is_down = False
            keypoints_to_draw = None
            if results[0].keypoints is not None:
                all_kps = results[0].keypoints.xy.cpu().numpy()
                closest_idx = np.argmin([np.linalg.norm(np.mean(kp, axis=0) - np.array([cx, cy])) for kp in all_kps])
                keypoints = all_kps[closest_idx]
                keypoints_to_draw = keypoints
                angle_turn, direction, turn_total = head_turn_angle(keypoints, TURN_THRESHOLD)
                angle_down, is_down, down_total = head_down_angle(keypoints, DOWN_THRESHOLD, direction is not None)
                person_stats[frame_id]["turn_time"] = turn_total
                person_stats[frame_id]["down_time"] = down_total

            if sleepy:
                state_text = "SLEEPING 😴"; color = (0, 0, 255)
            elif direction:
                state_text = f"LOOKING {direction.upper()}"; color = (0, 0, 255)
            elif is_down:
                state_text = "LOOKING DOWN"; color = (0, 140, 255)
            else:
                state_text = "FOCUSED"; color = (0, 255, 0)
            person_stats[frame_id]["state"] = state_text

            meeting_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "frame_id": frame_id,
                "frame_name": display_name,
                "state": state_text,
                "ear": round(ear_value, 3),
                "sleep_time": round(person_stats[frame_id]["sleep_time"], 2),
                "down_time": round(person_stats[frame_id]["down_time"], 2),
                "turn_time": round(person_stats[frame_id]["turn_time"], 2),
                "out_time": round(person_stats[frame_id]["out_time"], 2),
            })

            info_line = (f"F{frame_id} | Down:{person_stats[frame_id]['down_time']:.1f}s | "
                         f"Turn:{person_stats[frame_id]['turn_time']:.1f}s | "
                         f"Out:{person_stats[frame_id]['out_time']:.1f}s | "
                         f"Sleep:{person_stats[frame_id]['sleep_time']:.1f}s | "
                         f"EAR:{person_stats[frame_id]['ear']:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_name}: {state_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, info_line, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 🚪 Cảnh báo khung trống
    for frame_id, frame_info in spatial_frames.items():
        if frame_id not in occupied_frames:
            if now - frame_info["last_seen"] >= OUT_TIMEOUT:
                if not frame_info["is_empty"]:
                    frame_info["is_empty"] = True
                    frame_info["empty_start"] = now
                    print(f"🚪 Frame {frame_id} - Người rời khỏi khung hình")
                fx, fy = map(int, frame_info["center"])
                blink = int(now * BLINK_SPEED) % 2 == 0
                if blink:
                    cv2.rectangle(frame, (fx - WARNING_BOX_SIZE, fy - WARNING_BOX_SIZE),
                                  (fx + WARNING_BOX_SIZE, fy + WARNING_BOX_SIZE), (0, 0, 255), 3)
                display_name = person_names.get(frame_id, f"Frame {frame_id}")
                dur = now - frame_info["empty_start"]
                cv2.putText(frame, f"⚠️ {display_name}: ROI {dur:.1f}s", (fx - 100, fy - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Meeting Focus Monitor (Spatial)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if datetime.now() > end_time:
        break

# =====================================================
# 🔚 KẾT THÚC & LƯU FILE
# =====================================================
cap.release()
cv2.destroyAllWindows()
now_final = time.time()

for fid, info in spatial_frames.items():
    if info["is_empty"] and info["empty_start"]:
        person_stats[fid]["out_time"] += now_final - info["empty_start"]
    if person_stats.get(fid, {}).get("is_sleeping", False):
        sleep_duration = now_final - person_stats[fid]["sleep_start"]
        person_stats[fid]["sleep_time"] += sleep_duration
        person_stats[fid]["is_sleeping"] = False

total_session_time = now_final - session_start_timestamp
summary = []
for fid, stats in person_stats.items():
    sleep, down, turn, out = [stats.get(k, 0) for k in ("sleep_time", "down_time", "turn_time", "out_time")]
    total_distraction = sleep + down + turn + out
    rate = (total_distraction / total_session_time) * 100 if total_session_time > 0 else 0
    status = "❌ MẤT TẬP TRUNG" if rate > DISTRACTION_THRESHOLD * 100 else "✅ TẬP TRUNG TỐT"
    summary.append({
        "Frame_Name": person_names.get(fid, f"Frame {fid}"),
        "Frame_ID": fid,
        "Sleep_Time": round(sleep, 2),
        "Down_Time": round(down, 2),
        "Turn_Time": round(turn, 2),
        "Out_Time": round(out, 2),
        "Total_Distraction": round(total_distraction, 2),
        "Distraction_Rate_%": round(rate, 2),
        "Focus_Status": status,
        "Final_State": "OUT" if spatial_frames[fid]["is_empty"] else stats.get("state", "N/A"),
    })

pd.DataFrame(summary).to_csv(SUMMARY_FILE, index=False)
pd.DataFrame(meeting_log).to_csv(DETAIL_LOG_FILE, index=False)

print("\n📊 === MEETING SUMMARY ===")
print(pd.DataFrame(summary).to_string(index=False))
