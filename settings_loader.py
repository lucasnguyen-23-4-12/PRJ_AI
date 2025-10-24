# =====================================================
# ⚙️ settings_loader.py
# Thay cho settings.conf — chứa toàn bộ cấu hình cố định
# =====================================================

def load_settings():
    """
    Trả về dict chứa toàn bộ cấu hình của hệ thống Meeting Focus Monitor.
    Không cần đọc từ file .conf — cấu hình viết sẵn trong Python.
    """
    return {
        # 🎥 VIDEO & MODEL
        "VIDEO_SOURCE": 1,
        "MODEL_PATH": "./models/yolov8n-pose.pt",

        # ⏱️ THỜI GIAN
        "SESSION_DURATION": 180,       # 3 phút
        "OUT_TIMEOUT": 2.0,            # giây
        "SLEEP_CONFIRM_TIME": 5.0,     # giây

        # 📐 GÓC NHÌN
        "DOWN_THRESHOLD": 30,
        "TURN_THRESHOLD": 40,

        # 👁️ BUỒN NGỦ
        "EAR_SLEEP_THRESHOLD": 0.21,
        "EAR_FRAMES_CONFIRM": 5,

        # 📍 SPATIAL TRACKING
        "SPATIAL_TOLERANCE": 150,
        "FRAME_LEARNING_PERIOD": 30,

        # 🎨 HIỂN THỊ
        "WINDOW_WIDTH": 1280,
        "WINDOW_HEIGHT": 720,
        "WARNING_BOX_SIZE": 150,
        "BLINK_SPEED": 2,

        # 💾 LƯU FILE
        "SUMMARY_FILE": "meeting_summary.csv",
        "DETAIL_LOG_FILE": "meeting_detailed_log.csv",
    }
