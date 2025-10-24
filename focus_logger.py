import csv
from datetime import datetime
'''
class FocusLogger:
    def __init__(self, log_path="data/logs_focus.csv"):
        self.log_path = log_path
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "head_down", "head_turn", "eyes_closed"])
        self.head_down_count = 0
        self.head_turn_count = 0
        self.eye_closed_count = 0

    def log(self, head_down, head_turn, eyes_closed):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), head_down, head_turn, eyes_closed])
        if head_down: self.head_down_count += 1
        if head_turn: self.head_turn_count += 1
        if eyes_closed: self.eye_closed_count += 1

    def summary(self, limit_count=10):
        total = self.head_down_count + self.head_turn_count + self.eye_closed_count
        print("\n=== KẾT QUẢ SAU BUỔI HỌC ===")
        print(f"➡️ Cúi đầu: {self.head_down_count} lần")
        print(f"➡️ Quay lệch: {self.head_turn_count} lần")
        print(f"➡️ Nhắm mắt lâu: {self.eye_closed_count} lần")
        print(f"Tổng vi phạm: {total}")
        print("=> 🔴 MẤT TẬP TRUNG" if total > limit_count else "=> 🟢 TẬP TRUNG TỐT")
'''
from datetime import datetime

class FocusLogger:
    def __init__(self):
        self.head_down_count = 0
        self.head_turn_count = 0
        self.eye_closed_count = 0

    def log(self, head_down, head_turn, eyes_closed):
        if head_down: self.head_down_count += 1
        if head_turn: self.head_turn_count += 1
        if eyes_closed: self.eye_closed_count += 1

    def summary(self, limit_count=10):
        total = self.head_down_count + self.head_turn_count + self.eye_closed_count
        print("\n=== KẾT QUẢ SAU BUỔI HỌC ===")
        print(f"➡️ Cúi đầu: {self.head_down_count} lần")
        print(f"➡️ Quay lệch: {self.head_turn_count} lần")
        print(f"➡️ Nhắm mắt lâu: {self.eye_closed_count} lần")
        print(f"Tổng vi phạm: {total}")
        print("=> 🔴 MẤT TẬP TRUNG" if total > limit_count else "=> 🟢 TẬP TRUNG TỐT")

