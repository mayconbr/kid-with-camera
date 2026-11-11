import time
from collections import deque

import cv2
import numpy as np


class GestureDisappear:
    def __init__(self, webcam_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(webcam_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Não foi possível abrir a webcam. Verifique as permissões da câmera do macOS "
                "e o acesso do VS Code/Terminal."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.hidden = False
        self.last_toggle_time = 0.0
        self.toggle_cooldown = 1.6
        self.prev_skin_mask = None
        self.window_name = "Gesture Disappear"
        self.hold_until = 0.0
        self.motion_history = deque(maxlen=6)

    def _build_skin_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin_1 = np.array([0, 30, 50], dtype=np.uint8)
        upper_skin_1 = np.array([25, 255, 255], dtype=np.uint8)
        lower_skin_2 = np.array([170, 30, 50], dtype=np.uint8)
        upper_skin_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask_1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
        mask_2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _trigger_gesture(self, skin_mask):
        if self.prev_skin_mask is None:
            self.prev_skin_mask = skin_mask.copy()
            return False

        motion_mask = cv2.absdiff(skin_mask, self.prev_skin_mask)
        motion_ratio = cv2.countNonZero(motion_mask) / motion_mask.size
        skin_ratio = cv2.countNonZero(skin_mask) / skin_mask.size

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area

        self.prev_skin_mask = skin_mask.copy()
        self.motion_history.append(motion_ratio)

        if len(self.motion_history) < self.motion_history.maxlen:
            return False

        avg_motion = sum(self.motion_history) / len(self.motion_history)
        peak_motion = max(self.motion_history)

        trigger = (
            skin_ratio > 0.03
            and largest_area > 1500
            and peak_motion > avg_motion * 2.5
            and peak_motion > 0.006
        )
        return trigger

    def toggle_visibility(self):
        now = time.time()
        if now < self.hold_until:
            return

        self.hidden = not self.hidden
        self.last_toggle_time = now
        self.hold_until = now + self.toggle_cooldown

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        skin_mask = self._build_skin_mask(frame)

        if self._trigger_gesture(skin_mask):
            self.toggle_visibility()

        if self.hidden:
            frame[:] = 0

        return frame

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)
        cv2.moveWindow(self.window_name, 300, 120)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Não foi possível ler o frame da webcam.")
                    break

                processed = self.process_frame(frame)
                cv2.imshow(self.window_name, processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = GestureDisappear()
        app.run()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
