import os
import time
from collections import deque

import cv2
import numpy as np


def _load_cascade(filename):
    """Carrega um classificador Haar que acompanha o opencv-python."""
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, filename))


class GestureDisappear:
    """Detecta mao aberta/fechada na webcam e mostra o status na tela.

    Rostos sao detectados, destacados em outra cor (magenta) e removidos
    da mascara de pele, para que rosto/pescoco nao sejam confundidos com a mao.

    Estados possiveis (mostrados no topo da tela):
    - "sem mao": nenhuma mao detectada -> "MAO NAO DETECTADA"
    - "aberta": mao detectada com dedos abertos/afastados
    - "fechada": mao detectada fechada (punho)
    """

    STATE_NO_HAND = "sem mao"
    STATE_OPEN = "aberta"
    STATE_CLOSED = "fechada"

    STATE_LABELS = {
        STATE_NO_HAND: "MAO NAO DETECTADA",
        STATE_OPEN: "MAO ABERTA",
        STATE_CLOSED: "MAO FECHADA",
    }
    STATE_COLORS = {
        STATE_NO_HAND: (160, 160, 160),
        STATE_OPEN: (0, 255, 0),
        STATE_CLOSED: (80, 172, 255),
    }

    FACE_COLOR = (255, 0, 255)  # magenta (BGR) para as caixas de rosto

    # legenda do que cada cor desenhada no video significa (valores em BGR)
    LEGEND = [
        ((0, 0, 255), "ROI: regiao de deteccao da mao"),
        (FACE_COLOR, "Rosto detectado (ignorado)"),
        ((0, 255, 255), "Contorno da mao (pele detectada)"),
        ((255, 0, 0), "Caixa delimitadora da mao"),
        ((0, 255, 0), "Eixos do centro da caixa"),
    ]

    # ---- deteccao de rosto ----
    FACE_DETECT_WIDTH = 480    # largura da imagem reduzida usada na deteccao (mais rapida)
    FACE_SCALE_FACTOR = 1.1   # escala entre as passadas do cascade
    FACE_MIN_NEIGHBORS = 5     # vizinhos necessarios (mais alto = menos falsos positivos)
    FACE_MEMORY = 0.6          # segundos que um rosto fica lembrado sem ser redetectado

    # margens para APAGAR o rosto da mascara de pele (desconecta o rosto do
    # pescoco). Laterais pequenas para nao cortar uma mao ao lado do rosto.
    FACE_MASK_PAD_SIDE = 0.10   # fracao da largura do rosto
    FACE_MASK_PAD_TOP = 0.15    # fracao da altura do rosto
    FACE_MASK_PAD_BOTTOM = 0.40  # fracao da altura do rosto (queixo + inicio do pescoco)

    # zona de cabeca/pescoco: um blob LISO nessa regiao e tratado como corpo
    # (pescoco/ombros) e nao como mao fechada. Uma mao ABERTA, com vales
    # profundos entre os dedos, e aceita em qualquer lugar do quadro.
    HEAD_MARGIN_SIDE = 0.25     # margem lateral da cabeca (fracao da largura do rosto)
    HEAD_MARGIN_TOP = 0.35      # margem acima do rosto: cabelo/testa
    NECK_DEPTH = 0.9            # profundidade da zona abaixo do rosto: pescoco
    ZONE_BYPASS_DEFECTS = 3     # vais entre dedos p/ aceitar mao aberta dentro da zona

    # ---- parametros de deteccao da mao (ajuste se necessario) ----
    MIN_HAND_AREA = 1800        # area minima do contorno para considerar que ha uma mao
    HAND_MIN_ASPECT = 0.25      # proporcao largura/altura minima de uma mao
    HAND_MAX_ASPECT = 4.0       # proporcao largura/altura maxima (mais alongado = braco/pescoco)
    HISTORY_FRAMES = 8          # tamanho da janela deslizante de classificacoes recentes
    CONFIRM_VOTES = 5           # votos na janela para confirmar uma mudanca de estado
    NO_HAND_TIMEOUT = 0.4       # segundos sem mao para voltar ao estado "sem mao"
    OPEN_MIN_DEFECTS = 2        # qtd. de vales profundos entre os dedos -> mao aberta
    DEFECT_DEPTH_RATIO = 0.08   # profundidade minima do vale (fracao do tamanho da mao)
    OPEN_MAX_SOLIDITY = 0.78    # solidez abaixo disso -> mao aberta (punho fica acima de 0.85)

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
        self.face_cascade = _load_cascade("haarcascade_frontalface_alt2.xml")
        self.profile_cascade = _load_cascade("haarcascade_profileface.xml")
        self._reset_tracking()

    def _reset_tracking(self):
        """(Re)inicia o estado do rastreamento de mao e rosto."""
        self.gesture_state = self.STATE_NO_HAND
        self.last_hand_time = 0.0
        self._history = deque(maxlen=self.HISTORY_FRAMES)
        self._candidate = None
        self._candidate_votes = 0
        self._tracked_faces = []

    # ------------------------------------------------------- deteccao de rosto

    def _detect_faces(self, frame):
        """Procura rostos no frame e devolve caixas (x, y, w, h).

        Usa Haar cascades numa versao reduzida do frame (mais rapido).
        Detecta rosto de frente e de perfil (nos dois lados).
        """
        if self.face_cascade.empty() and self.profile_cascade.empty():
            return []

        scale = self.FACE_DETECT_WIDTH / frame.shape[1]
        small_w = self.FACE_DETECT_WIDTH
        small_h = max(1, int(round(frame.shape[0] * scale)))
        small = cv2.resize(frame, (small_w, small_h))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        kwargs = {
            "scaleFactor": self.FACE_SCALE_FACTOR,
            "minNeighbors": self.FACE_MIN_NEIGHBORS,
            "minSize": (36, 36),
        }

        rects = []
        if not self.face_cascade.empty():
            rects.extend(self.face_cascade.detectMultiScale(gray, **kwargs))
        if not self.profile_cascade.empty():
            rects.extend(self.profile_cascade.detectMultiScale(gray, **kwargs))
            flipped = cv2.flip(gray, 1)
            for (fx, fy, fw, fh) in self.profile_cascade.detectMultiScale(flipped, **kwargs):
                # espelha o x de volta para o sentido original
                rects.append((small_w - int(fx) - int(fw), int(fy), int(fw), int(fh)))

        boxes = []
        for (fx, fy, fw, fh) in rects:
            box = (
                int(round(fx / scale)),
                int(round(fy / scale)),
                int(round(fw / scale)),
                int(round(fh / scale)),
            )
            # descarta duplicatas: o mesmo rosto achado por cascades diferentes
            if any(self._iou(box, other) > 0.4 for other in boxes):
                continue
            boxes.append(box)
        return boxes

    @staticmethod
    def _iou(a, b):
        """Intersecao sobre uniao de duas caixas no formato (x, y, w, h)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _track_faces(self, detected, now):
        """Mantem um rosto por FACE_MEMORY segundos mesmo quando o detector
        falha por alguns frames (evita que a exclusao do rosto pisque).
        """
        remaining = list(detected)
        tracked = []
        for face in self._tracked_faces:
            match = None
            for i, box in enumerate(remaining):
                if self._iou(face["box"], box) > 0.25:
                    match = i
                    break
            if match is not None:
                box = remaining.pop(match)
                tracked.append({"box": box, "last_seen": now})
            elif now - face["last_seen"] < self.FACE_MEMORY:
                tracked.append(face)
        for box in remaining:
            tracked.append({"box": box, "last_seen": now})
        self._tracked_faces = tracked
        return [face["box"] for face in tracked]

    def _remove_faces_from_mask(self, mask, faces, roi_x, roi_y):
        """Apaga da mascara de pele as regioes de rosto (com margem para
        queixo e pescoco), para o rosto nao virar 'mao'."""
        for (fx, fy, fw, fh) in faces:
            pad_side = int(fw * self.FACE_BOX_EXPAND) // 2
            pad_top = int(fh * self.FACE_BOX_EXPAND) // 2
            pad_bottom = int(fh * self.FACE_BOX_EXPAND)
            x1 = max(0, fx - pad_side - roi_x)
            y1 = max(0, fy - pad_top - roi_y)
            x2 = min(mask.shape[1], fx + fw + pad_side - roi_x)
            y2 = min(mask.shape[0], fy + fh + pad_bottom - roi_y)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    # ------------------------------------------------------------- deteccao da mao

    def _select_hand(self, contours, faces, roi_x, roi_y):
        """Escolhe o contorno da mao entre os contornos de pele.

        Regras, em ordem:
        - testa do maior contorno para o menor;
        - o contorno precisa parecer uma mao (area e proporcao);
        - nao pode estar colado na regiao do rosto/cabeca: pescoco, ombros,
          peito e cabelo ficam nessa zona proibida e nao sao maos.
        """
        zones = []
        for (fx, fy, fw, fh) in faces:
            zones.append((
                fx - int(fw * self.HEAD_ZONE_SIDE) - roi_x,
                fy - int(fh * self.HEAD_ZONE_TOP) - roi_y,
                fx + fw + int(fw * self.HEAD_ZONE_SIDE) - roi_x,
                fy + fh + int(fh * self.HEAD_ZONE_BOTTOM) - roi_y,
            ))

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            metrics = self._hand_metrics(contour)
            if metrics is None:
                continue
            bx, by, bw, bh = metrics["bbox"]
            colada_na_cabeca = any(
                bx < zx2 and bx + bw > zx1 and by < zy2 and by + bh > zy1
                for (zx1, zy1, zx2, zy2) in zones
            )
            if colada_na_cabeca:
                continue
            return contour, metrics
        return None, None

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

        Retorna um dict com as metricas ou None se o contorno nao parecer uma mao.
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
        aspect = float(w) / float(h)
        if aspect < self.HAND_MIN_ASPECT or aspect > self.HAND_MAX_ASPECT:
            return None  # alongado demais: provavelmente braco, pescoco ou corpo

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
        """Maquina de estados com janela deslizante (anti-ruido).

        O estado so muda quando a janela dos ultimos HISTORY_FRAMES frames
        acumula CONFIRM_VOTES classificacoes apontando para o estado novo.
        Assim um frame isolado ruidoso nao zera a contagem (o que antes
        deixava a mudanca de status travada em "confirmando").
        """
        if metrics is None:
            if now - self.last_hand_time > self.NO_HAND_TIMEOUT:
                self.gesture_state = self.STATE_NO_HAND
                self._history.clear()
                self._candidate = None
                self._candidate_votes = 0
            return

        self.last_hand_time = now
        classification = self.STATE_OPEN if metrics["is_open"] else self.STATE_CLOSED
        self._history.append(classification)

        if classification == self.gesture_state:
            # frame concorda com o estado atual: nada a confirmar agora,
            # mas os votos anteriores da janela continuam valendo
            self._candidate = None
            self._candidate_votes = 0
            return

        votes = sum(1 for c in self._history if c == classification)
        self._candidate = classification
        self._candidate_votes = votes
        if votes < self.CONFIRM_VOTES:
            return

        self.gesture_state = classification
        self._history.clear()
        self._candidate = None
        self._candidate_votes = 0

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
                f"confirmando: {self.STATE_LABELS[self._candidate]} ({self._candidate_votes}/{self.CONFIRM_VOTES})",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_faces(self, frame, faces):
        """Caixas magenta com a label ROSTO, para diferenciar da mao."""
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), self.FACE_COLOR, 2)
            cv2.putText(
                frame,
                "ROSTO",
                (fx + 6, fy + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.FACE_COLOR,
                2,
                cv2.LINE_AA,
            )

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        now = time.time()

        roi_x, roi_y, roi_w, roi_h = self._roi(frame)
        faces = self._track_faces(self._detect_faces(frame), now)

        # a deteccao da mao usa o frame original, antes de qualquer desenho;
        # as regioes de rosto sao removidas para nao virarem "mao"
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        mask = self._skin_mask(roi)
        if faces:
            mask = self._remove_faces_from_mask(mask, faces, roi_x, roi_y)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour, metrics = self._select_hand(contours, faces, roi_x, roi_y)

        self._update_state(metrics, now)

        # ---- desenhos (veja a legenda no canto inferior esquerdo) ----
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        self._draw_faces(frame, faces)

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
