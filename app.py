import time

import cv2
import numpy as np


class GestureDisappear:
    """Detecta mao aberta/fechada na webcam, mostra o status na tela e
    dispara um "OK" quando a mao abre e depois fecha.

    Estados possiveis (mostrados no topo da tela):
    - "sem mao": nenhuma mao detectada na regiao de interesse (ROI)
    - "aberta": mao detectada com dedos abertos/afastados
    - "fechada": mao detectada fechada (punho)
    """

    STATE_NO_HAND = "sem mao"
    STATE_OPEN = "aberta"
    STATE_CLOSED = "fechada"

    STATE_LABELS = {
        STATE_NO_HAND: "SEM MAO",
        STATE_OPEN: "MAO ABERTA",
        STATE_CLOSED: "MAO FECHADA",
    }
    STATE_COLORS = {
        STATE_NO_HAND: (160, 160, 160),
        STATE_OPEN: (0, 255, 0),
        STATE_CLOSED: (80, 172, 255),
    }

    # legenda do que cada cor desenhada no video significa (valores em BGR)
    LEGEND = [
        ((0, 0, 255), "ROI: regiao de deteccao da mao"),
        ((0, 255, 255), "Contorno da mao (pele detectada)"),
        ((255, 0, 0), "Caixa delimitadora da mao"),
        ((0, 255, 0), "Eixos do centro da caixa"),
    ]

    # ---- parametros de deteccao (ajuste se necessario) ----
    MIN_HAND_AREA = 1800        # area minima do contorno para considerar que ha uma mao
    MIN_CONFIRM_FRAMES = 4     # frames consecutivos para confirmar uma mudanca de estado
    NO_HAND_TIMEOUT = 0.4       # segundos sem mao para voltar ao estado "sem mao"
    FEEDBACK_DURATION = 0.8     # segundos que o "OK" fica na tela
    OPEN_MIN_DEFECTS = 2       # qtd. de vales profundos entre os dedos -> mao aberta
    DEFECT_DEPTH_RATIO = 0.08  # profundidade minima do vale (fracao do tamanho da mao)
    OPEN_MAX_SOLIDITY = 0.78   # solidez abaixo disso -> mao aberta (punho fica acima de 0.85)

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
        self.gesture_state = self.STATE_NO_HAND
        self.last_hand_time = 0.0
        self.last_trigger_time = -10.0
        self._candidate = None
        self._candidate_count = 0

    # ------------------------------------------------------------- deteccao

    @staticmethod
    def _skin_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 25, 50], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    @staticmethod
    def _roi(frame):
        """Regiao de interesse (ROI) onde a mao e procurada."""
        roi_x = int(frame.shape[1] * 0.08)
        roi_y = int(frame.shape[0] * 0.02)
        roi_w = int(frame.shape[1] * 0.84)
        roi_h = int(frame.shape[0] * 0.94)
        return roi_x, roi_y, roi_w, roi_h

    def _hand_metrics(self, contour):
        """Classifica o contorno como mao aberta ou fechada.

        Retorna um dict com as metricas ou None se o contorno for pequeno demais.
        - solidez: area do contorno / area do casco convexo. Punho fechado fica
          proximo de 1.0; mao aberta com dedos afastados fica bem abaixo.
        - vales profundos (convexity defects): espacos entre os dedos. Mao aberta
          tem varios; punho fechado praticamente nenhum.
        """
        area = cv2.contourArea(contour)
        if area < self.MIN_HAND_AREA:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 1.0

        deep_defects = 0
        try:
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                defects = cv2.convexityDefects(contour, hull_idx)
                if defects is not None:
                    ref = float(max(w, h))
                    for i in range(defects.shape[0]):
                        depth = defects[i, 0][3] / 256.0
                        if depth > ref * self.DEFECT_DEPTH_RATIO:
                            deep_defects += 1
        except cv2.error:
            pass

        is_open = deep_defects >= self.OPEN_MIN_DEFECTS or solidity < self.OPEN_MAX_SOLIDITY
        return {
            "area": area,
            "bbox": (x, y, w, h),
            "solidity": solidity,
            "deep_defects": deep_defects,
            "is_open": is_open,
        }

    def _update_state(self, metrics, now):
        """Maquina de estados com anti-tremido (debounce).

        A mudanca de estado so acontece depois de MIN_CONFIRM_FRAMES frames
        consecutivos com a mesma classificacao.
        """
        if metrics is None:
            if now - self.last_hand_time > self.NO_HAND_TIMEOUT:
                self.gesture_state = self.STATE_NO_HAND
                self._candidate = None
                self._candidate_count = 0
            return

        self.last_hand_time = now
        target = self.STATE_OPEN if metrics["is_open"] else self.STATE_CLOSED

        if target == self.gesture_state:
            self._candidate = None
            self._candidate_count = 0
            return

        if target == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = target
            self._candidate_count = 1

        if self._candidate_count < self.MIN_CONFIRM_FRAMES:
            return

        previous, self.gesture_state = self.gesture_state, target
        self._candidate = None
        self._candidate_count = 0

        if previous == self.STATE_OPEN and target == self.STATE_CLOSED:
            if now - self.last_trigger_time > self.FEEDBACK_DURATION:
                self.last_trigger_time = now  # abriu e fechou: dispara o "OK"

    # -------------------------------------------------------------- desenho

    def _draw_legend(self, frame):
        line_h = 26
        x0 = 18
        y0 = frame.shape[0] - 18 - len(self.LEGEND) * line_h
        top = max(0, y0 - 14)
        bottom = min(frame.shape[0], y0 + len(self.LEGEND) * line_h)
        left = max(0, x0 - 12)
        right = min(frame.shape[1], x0 + 440)
        if bottom > top and right > left:
            panel = frame[top:bottom, left:right]
            frame[top:bottom, left:right] = cv2.addWeighted(panel, 0.45, np.zeros_like(panel), 0.55, 0)

        for i, (color, text) in enumerate(self.LEGEND):
            y = y0 + i * line_h
            cv2.line(frame, (x0, y + 9), (x0 + 30, y + 9), color, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x0 + 40, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_status(self, frame, metrics):
        label = self.STATE_LABELS[self.gesture_state]
        color = self.STATE_COLORS[self.gesture_state]
        cv2.putText(frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        if metrics is not None:
            cv2.putText(
                frame,
                f"area={int(metrics['area'])} solidez={metrics['solidity']:.2f} vales={metrics['deep_defects']}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        if self._candidate is not None:
            cv2.putText(
                frame,
                f"confirmando: {self.STATE_LABELS[self._candidate]} ({self._candidate_count}/{self.MIN_CONFIRM_FRAMES})",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_feedback(self, frame, now):
        if 0 <= now - self.last_trigger_time <= self.FEEDBACK_DURATION:
            cv2.putText(
                frame,
                "OK",
                (frame.shape[1] // 2 - 90, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        now = time.time()

        roi_x, roi_y, roi_w, roi_h = self._roi(frame)

        # a deteccao usa o frame original, antes de qualquer desenho
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        mask = self._skin_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = None
        metrics = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            metrics = self._hand_metrics(largest)
            if metrics is not None:
                contour = largest

        self._update_state(metrics, now)

        # ---- desenhos (veja a legenda no canto inferior esquerdo) ----
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

        if contour is not None:
            x, y, w, h = metrics["bbox"]
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2, offset=(roi_x, roi_y))
            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + roi_x + w, y + roi_y + h), (255, 0, 0), 2)
            cx = x + roi_x + w // 2
            cy = y + roi_y + h // 2
            cv2.line(frame, (cx, y + roi_y), (cx, y + roi_y + h), (0, 255, 0), 2)
            cv2.line(frame, (x + roi_x, cy), (x + roi_x + w, cy), (0, 255, 0), 2)

        self._draw_status(frame, metrics)
        self._draw_legend(frame)
        self._draw_feedback(frame, now)
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
