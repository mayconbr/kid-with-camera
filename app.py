import time

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

        self.window_name = "Gesture Disappear"
        self.last_trigger_time = 0.0
        self.feedback_duration = 0.8
        self.gesture_state = "idle"
        self.last_state_change = 0.0
        self.last_open_area = 0.0

    @staticmethod
    def _skin_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 25, 50], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def detect_open_then_close(self, frame):
        roi_x = int(frame.shape[1] * 0.08)
        roi_y = int(frame.shape[0] * 0.02)
        roi_w = int(frame.shape[1] * 0.84)
        roi_h = int(frame.shape[0] * 0.94)

        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        mask = self._skin_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.gesture_state = "idle"
            self.last_open_area = 0.0
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 1200:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        if w <= 0 or h <= 0:
            return None

        aspect = float(w) / float(h)
        now = time.time()

        if self.gesture_state == "idle":
            if area > 1800 and aspect > 0.6:
                self.gesture_state = "open"
                self.last_open_area = area
                self.last_state_change = now
            return None

        if self.gesture_state == "open":
            if area < self.last_open_area * 0.75 and now - self.last_state_change > 0.25:
                if now - self.last_trigger_time > self.feedback_duration:
                    self.last_trigger_time = now
                    self.gesture_state = "idle"
                    self.last_open_area = 0.0
                    self.last_state_change = now
                    return "OK"

            if area > self.last_open_area * 1.25:
                self.last_open_area = area
                self.last_state_change = now

        return None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)

        roi_x = int(frame.shape[1] * 0.08)
        roi_y = int(frame.shape[0] * 0.02)
        roi_w = int(frame.shape[1] * 0.84)
        roi_h = int(frame.shape[0] * 0.94)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        mask = self._skin_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest], -1, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + roi_x + w, y + roi_y + h), (255, 0, 0), 2)
            cv2.line(frame, (x + roi_x + w // 2, y + roi_y), (x + roi_x + w // 2, y + roi_y + h), (0, 255, 0), 2)
            cv2.line(frame, (x + roi_x, y + roi_y + h // 2), (x + roi_x + w, y + roi_y + h // 2), (0, 255, 0), 2)

        overlay_text = self.detect_open_then_close(frame)
        cv2.putText(
            frame,
            f"state={self.gesture_state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if overlay_text:
            cv2.putText(
                frame,
                overlay_text,
                (frame.shape[1] // 2 - 90, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )

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
