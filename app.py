import math
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
    da mascara de pele. A distincao entre mao e corpo e feita pelo FORMATO
    do contorno: mao aberta (vales entre os dedos) e aceita em qualquer
    lugar do quadro; um blob liso (punho/pescoco/ombro) so e aceito fora
    da regiao da cabeca/pescoco.

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
    ZONE_BYPASS_DEFECTS = 3     # vales entre dedos p/ aceitar mao aberta dentro da zona

    # ---- parametros de deteccao da mao (ajuste se necessario) ----
    MIN_HAND_AREA = 1800        # area minima do contorno para considerar que ha uma mao
    HAND_MIN_ASPECT = 0.25      # proporcao largura/altura minima de uma mao
    HAND_MAX_ASPECT = 4.0       # proporcao largura/altura maxima (mais alongado = braco/pescoco)
    HISTORY_FRAMES = 8          # tamanho da janela deslizante de classificacoes recentes
    CONFIRM_VOTES = 5           # votos na janela para confirmar uma mudanca de estado
    NO_HAND_TIMEOUT = 0.4       # segundos sem mao para voltar ao estado "sem mao"
    OPEN_MIN_DEFECTS = 2        # vales entre dedos necessarios para considerar mao aberta
    DEFECT_DEPTH_RATIO = 0.08   # profundidade minima do vale (fracao do tamanho da mao)
    DEFECT_MAX_CHORD = 0.22     # largura maxima do vale entre dedos (fracao do tamanho da
                                # mao). Vales mais largos sao axila/cotovelo/braco, nao
                                # espacos entre dedos

    PIKACHU_SECONDS = 2.6       # tempo que o Pikachu fica na tela ao fechar a mao

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
        self.pikachu_until = 0.0
        self._history = deque(maxlen=self.HISTORY_FRAMES)
        self._candidate = None
        self._candidate_votes = 0
        self._last_rejection = None
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
        queixo e pescoco), para o rosto nao virar 'mao'.

        As margens laterais sao pequenas de proposito: cortar demais
        apagaria uma mao posicionada logo ao lado do rosto.
        """
        for (fx, fy, fw, fh) in faces:
            pad_side = int(fw * self.FACE_MASK_PAD_SIDE)
            pad_top = int(fh * self.FACE_MASK_PAD_TOP)
            pad_bottom = int(fh * self.FACE_MASK_PAD_BOTTOM)
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

        A distincao entre mao e corpo e feita principalmente pelo FORMATO
        do contorno, nao pela posicao:

        - um contorno com vales profundos entre os dedos (mao aberta) e
          aceito em qualquer lugar do quadro, ate colado no rosto;
        - um contorno LISO (sem vales), como punho, pescoco, ombro ou
          cabelo, so e aceito FORA da regiao da cabeca/pescoco.

        Devolve (contorno, metricas, motivo). O motivo e None quando a
        mao foi encontrada; quando nao ha mao, explica por que o maior
        blob de pele foi descartado.
        """
        zones = []
        for (fx, fy, fw, fh) in faces:
            side = int(fw * self.HEAD_MARGIN_SIDE)
            zones.append((
                fx - side - roi_x,
                fy - int(fh * self.HEAD_MARGIN_TOP) - roi_y,
                fx + fw + side - roi_x,
                fy + fh + int(fh * self.NECK_DEPTH) - roi_y,
            ))

        rejection = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            metrics = self._hand_metrics(contour)
            if metrics is None:
                continue  # pequeno ou alongado demais (ruido, braco)
            bx, by, bw, bh = metrics["bbox"]
            na_zona = any(
                bx < zx2 and bx + bw > zx1 and by < zy2 and by + bh > zy1
                for (zx1, zy1, zx2, zy2) in zones
            )
            if na_zona and metrics["deep_defects"] < self.ZONE_BYPASS_DEFECTS:
                # blob liso dentro da zona cabeca/pescoco: e corpo, nao mao
                rejection = "blob liso na zona do rosto/pescoco"
                continue
            return contour, metrics, None
        return None, None, rejection

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
        - solidez: area do contorno / area do casco convexo (informativa; um
          braco dobrado tambem tem solidez baixa, entao ela NAO decide sozinha).
        - vais entre dedos (convexity defects): espacos entre os dedos. Mao
          aberta tem varios; punho fechado praticamente nenhum. Um vale so
          conta como "vale de dedo" se for profundo E estreito (corda curta):
          bracos/troncos tem cavidades profundas, mas largas (axila, cotovelo).
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
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth <= ref * self.DEFECT_DEPTH_RATIO:
                            continue  # raso demais
                        start = contour[s][0]
                        end = contour[e][0]
                        chord = float(np.hypot(end[0] - start[0], end[1] - start[1]))
                        if chord > ref * self.DEFECT_MAX_CHORD:
                            continue  # largo demais: axila/cotovelo/braco
                        deep_defects += 1
        except cv2.error:
            pass

        is_open = deep_defects >= self.OPEN_MIN_DEFECTS
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
        if self.gesture_state == self.STATE_CLOSED:
            self.pikachu_until = now + self.PIKACHU_SECONDS

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

    def _draw_pikachu(self, frame, shoulder=None, scale=1.0, elapsed=None):
        """Desenha um Pikachu 3D estilo anime no ombro.

        Cada parte recebe um gradiente (luz vinda de cima e da esquerda) e o
        personagem projeta uma sombra suave no video, dando volume. Quando
        `elapsed` (segundos desde o aparecimento) e informado, ele entra com
        um "pop", balanca o corpo, mexe orelhas/cauda e pisca de vez em
        quando. `elapsed=None` desenha a pose parada (util nos testes).
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if shoulder is None:
            shoulder = (int(w * 0.82), int(h * 0.72))
        sxp, syp = shoulder
        s = float(scale)

        # ---- animacao: pop de entrada, balanco, orelhas, cauda e piscada ----
        if elapsed is None:
            pop, bob, ear_sway, tail_sway, blink = 1.0, 0.0, 0.0, 0.0, False
        else:
            e = max(0.0, float(elapsed))
            t = min(1.0, e / 0.45)
            c1, c3 = 1.70158, 2.70158
            tm = t - 1.0
            pop = max(0.0, 1.0 + c3 * tm ** 3 + c1 * tm ** 2)  # ease-out-back
            bob = -abs(math.sin(e * 6.2)) * 6.0                # px
            ear_sway = math.sin(e * 5.0) * 5.0                 # graus
            tail_sway = math.sin(e * 3.0) * 10.0               # graus
            blink = (e % 1.15) > 1.06

        # ---- paleta (BGR) e gradientes de cada parte: (claro, escuro) ----
        OUTLINE = (40, 64, 96)
        BLACK = (26, 26, 26)
        WHITE = (255, 255, 255)
        ROLES = {
            "yellow": ((126, 242, 255), (8, 138, 186)),
            "cheek": ((96, 132, 255), (26, 44, 200)),
            "black": ((58, 58, 58), (12, 12, 12)),
            "brown": ((96, 158, 218), (30, 76, 132)),
        }

        # ---- sprite local onde o Pikachu e montado (1 px = 1 px na tela) ----
        ppu = s                            # pixels do sprite por unidade do desenho
        OX, OY = 258.0, 362.0              # deslocamento para manter tudo positivo
        HP = 1.12                          # folga para o "pop" passar de 1.0
        WW = int(round((245.0 * HP + OX) * ppu)) + 8
        HH = int(round((160.0 + 75.0 * HP + OY) * ppu)) + 8
        canvas = np.zeros((HH, WW, 3), np.uint8)
        mask = np.zeros((HH, WW), np.uint8)  # silhueta (alpha) para a sombra

        PIVOT_Y = 160.0

        def P(dx, dy):
            px = dx * pop + OX
            py = PIVOT_Y + (dy - PIVOT_Y) * pop + OY
            return (int(round(px * ppu)), int(round(py * ppu)))

        def L(v):
            return max(1, int(round(v * ppu * pop)))

        def A(a):
            return (max(1, int(round(a[0] * ppu * pop))),
                    max(1, int(round(a[1] * ppu * pop))))

        def rot(point, base, deg):
            """Gira um ponto (unidades do desenho) em torno de `base`."""
            ang = math.radians(deg)
            ca, sa = math.cos(ang), math.sin(ang)
            x, y = point[0] - base[0], point[1] - base[1]
            return (base[0] + x * ca - y * sa, base[1] + x * sa + y * ca)

        def apply_local(pm, bb, role, fill):
            """Pinta o gradiente 3D dentro de `pm` (recorte `bb`) no canvas."""
            x0, y0, x1, y1 = bb
            sub = pm > 0
            region = canvas[y0:y1, x0:x1]
            if role is not None:
                hi, lo = ROLES[role]
                hh, ww = pm.shape
                ny = (np.arange(hh, dtype=np.float32) - (hh - 1) * 0.5) / max(1.0, (hh - 1) * 0.5)
                nx = (np.arange(ww, dtype=np.float32) - (ww - 1) * 0.5) / max(1.0, (ww - 1) * 0.5)
                # luz de cima-esquerda: claro no alto/esquerda, escuro embaixo/direita
                lit = np.clip(0.56 - 0.30 * nx[None, :] - 0.38 * ny[:, None],
                              0.0, 1.0) ** 0.85
                for ch in range(3):
                    vals = lo[ch] + (hi[ch] - lo[ch]) * lit
                    region[..., ch][sub] = vals[sub].astype(np.uint8)
            else:
                region[sub] = fill
            mreg = mask[y0:y1, x0:x1]
            np.maximum(mreg, pm, out=mreg)

        def clip_bb(x0, y0, x1, y1):
            return (max(0, int(x0)), max(0, int(y0)),
                    min(WW, int(x1) + 1), min(HH, int(y1) + 1))

        def ellipse_part(center, axes, angle=0, role="yellow", fill=None,
                         outline=OUTLINE, thickness=4):
            c, ax = P(*center), A(axes)
            th = L(thickness) if outline is not None and thickness else 0
            r = max(ax) + th + 2
            bb = clip_bb(c[0] - r, c[1] - r, c[0] + r, c[1] + r)
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                return
            pm = np.zeros((bb[3] - bb[1], bb[2] - bb[0]), np.uint8)
            cv2.ellipse(pm, (c[0] - bb[0], c[1] - bb[1]), ax, angle, 0, 360,
                        255, -1, cv2.LINE_AA)
            apply_local(pm, bb, role, fill)
            if th:
                cv2.ellipse(canvas, c, ax, angle, 0, 360, outline, th, cv2.LINE_AA)
                cv2.ellipse(mask, c, ax, angle, 0, 360, 255, th, cv2.LINE_AA)

        def poly_part(points, role="yellow", fill=None, outline=OUTLINE,
                      thickness=4, closed=True):
            pts = np.array([P(*p) for p in points], np.int32)
            th = L(thickness) if outline is not None and thickness else 0
            pad = th + 2
            bb = clip_bb(pts[:, 0].min() - pad, pts[:, 1].min() - pad,
                         pts[:, 0].max() + pad, pts[:, 1].max() + pad)
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                return
            pm = np.zeros((bb[3] - bb[1], bb[2] - bb[0]), np.uint8)
            local = (pts - (bb[0], bb[1])).reshape((-1, 1, 2))
            cv2.fillPoly(pm, [local], 255, cv2.LINE_AA)
            apply_local(pm, bb, role, fill)
            if th:
                glob = pts.reshape((-1, 1, 2))
                cv2.polylines(canvas, [glob], closed, outline, th, cv2.LINE_AA)
                cv2.polylines(mask, [glob], closed, 255, th, cv2.LINE_AA)

        def stroke_part(points, width, role="yellow", outline=OUTLINE, pad=4.5):
            """Traco grosso (tubo) com gradiente, para a cauda de raio."""
            pts = np.array([P(*p) for p in points], np.int32)
            th_out = L(width + 2 * pad) if outline is not None else 0
            bb = clip_bb(pts[:, 0].min() - max(th_out, L(width)) // 2 - 3,
                         pts[:, 1].min() - max(th_out, L(width)) // 2 - 3,
                         pts[:, 0].max() + max(th_out, L(width)) // 2 + 3,
                         pts[:, 1].max() + max(th_out, L(width)) // 2 + 3)
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                return
            glob = pts.reshape((-1, 1, 2))
            if th_out:
                cv2.polylines(canvas, [glob], False, outline, th_out, cv2.LINE_AA)
                cv2.polylines(mask, [glob], False, 255, th_out, cv2.LINE_AA)
            pm = np.zeros((bb[3] - bb[1], bb[2] - bb[0]), np.uint8)
            local = (pts - (bb[0], bb[1])).reshape((-1, 1, 2))
            cv2.polylines(pm, [local], False, 255, L(width), cv2.LINE_AA)
            apply_local(pm, bb, role, fill=None)

        def dot(center, radius, color):
            c, r = P(*center), max(1, int(round(radius * ppu * pop)))
            cv2.circle(canvas, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(mask, c, r, 255, -1, cv2.LINE_AA)

        def arc(center, axes, start, end, color, thickness):
            c, ax, th = P(*center), A(axes), L(thickness)
            cv2.ellipse(canvas, c, ax, 0, start, end, color, th, cv2.LINE_AA)
            cv2.ellipse(mask, c, ax, 0, start, end, 255, th, cv2.LINE_AA)

        # ---- cauda de raio (atras do corpo), girando em torno da base ----
        tail_base = (84.0, 100.0)
        def t(p):
            return rot(p, tail_base, tail_sway)

        stroke_part([t((84, 100)), t((106, 88))], 30, role="brown")
        band = [t((104, 86)), t((130, 62)), t((102, 36)), t((146, -6))]
        stroke_part(band, 21)
        # ponta larga e achatada caracteristica
        ux, uy = 0.723, -0.690
        nx, ny = 0.690, 0.723
        tip_len, tip_half = 34.0, 36.0
        bx, by = 146.0, -6.0
        quad = [
            t((bx + nx * tip_half, by + ny * tip_half)),
            t((bx - nx * tip_half, by - ny * tip_half)),
            t((bx + ux * tip_len - nx * tip_half, by + uy * tip_len - ny * tip_half)),
            t((bx + ux * tip_len + nx * tip_half, by + uy * tip_len + ny * tip_half)),
        ]
        poly_part(quad, thickness=4)

        # ---- pes com tres dedos ----
        for foot_x in (-54, 54):
            ellipse_part((foot_x, 156), (58, 32))
            for toe in (-16, 0, 16):
                p1, p2 = P(foot_x + toe, 160), P(foot_x + toe - 3, 179)
                cv2.line(canvas, p1, p2, OUTLINE, L(3), cv2.LINE_AA)
                cv2.line(mask, p1, p2, 255, L(3), cv2.LINE_AA)

        # ---- corpo ----
        ellipse_part((0, 84), (88, 80))

        # ---- bracos curtos com maos redondas ----
        ellipse_part((-86, 52), (38, 21), angle=50)
        ellipse_part((86, 52), (38, 21), angle=-50)
        ellipse_part((-112, 21), (16, 15))
        ellipse_part((112, 21), (16, 15))

        # ---- orelhas compridas com ponta preta (atras da cabeca) ----
        for base, ear, tip, swing in (
            ((-70, -112), [(-34, -118), (-112, -250), (-80, -108)],
             [(-82, -200), (-100, -196), (-112, -250)], ear_sway),
            ((70, -112), [(34, -118), (112, -250), (80, -108)],
             [(82, -200), (100, -196), (112, -250)], -ear_sway),
        ):
            ear = [rot(p, base, swing) for p in ear]
            tip = [rot(p, base, swing) for p in tip]
            poly_part(ear, outline=None)
            poly_part(tip, role="black", outline=None)
            ear_pts = np.array([P(*p) for p in ear], np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [ear_pts], True, OUTLINE, L(4), cv2.LINE_AA)
            cv2.polylines(mask, [ear_pts], True, 255, L(4), cv2.LINE_AA)
        # ---- cabeca grande ----
        ellipse_part((0, -66), (114, 96))

        # ---- olhos grandes com brilho (fechados na piscada) ----
        for eye_x in (-42, 42):
            if blink:
                arc((eye_x, -70), (20, 16), 20, 160, BLACK, 4)
            else:
                ellipse_part((eye_x, -76), (20, 25), role="black", outline=None)
                dot((eye_x - 5, -85), 8, WHITE)
                dot((eye_x + 7, -66), 3.6, WHITE)

        # ---- bochechas vermelhas ----
        ellipse_part((-84, -26), (23, 23), role="cheek", outline=None)
        ellipse_part((84, -26), (23, 23), role="cheek", outline=None)

        # ---- nariz minusculo ----
        poly_part([(-6, -46), (6, -46), (0, -38)], role=None, fill=BLACK,
                  outline=None, thickness=0)

        # ---- sorriso em "w" (dois arcos que se encontram no centro) ----
        for mouth_x in (-23, 23):
            arc((mouth_x, -24), (24, 16), 0, 180, BLACK, 4)

        # ---- sombra projetada (silhueta desfocada, deslocada p/ baixo/direita) ----
        # Calculada em resolucao reduzida: a sombra e suave, nao precisa de detalhe.
        q = 4.0
        sqw, sqh = max(1, int(WW / q)), max(1, int(HH / q))
        small_mask = cv2.resize(mask, (sqw, sqh), interpolation=cv2.INTER_AREA)
        sh = cv2.GaussianBlur(small_mask.astype(np.float32), (0, 0),
                              max(1.0, 4.5 * ppu / q))
        shift = np.float32([[1, 0, 9.0 * s / q], [0, 1, 11.0 * s / q]])
        shadow = cv2.warpAffine(sh, shift, (sqw, sqh), flags=cv2.INTER_LINEAR,
                                borderValue=0)

        # ---- usa o sprite diretamente (ja esta no tamanho final) ----
        W2, H2 = WW, HH
        sprite = canvas
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        shad = cv2.resize(shadow / 255.0, (W2, H2),
                          interpolation=cv2.INTER_LINEAR)[..., None]

        x0 = int(round(sxp - OX * s))
        y0 = int(round(syp + bob - OY * s))
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, x0 + W2), min(h, y0 + H2)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        rx0, ry0 = fx0 - x0, fy0 - y0
        rw, rh = fx1 - fx0, fy1 - fy0

        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        a = alpha[ry0:ry0 + rh, rx0:rx0 + rw]
        sd = shad[ry0:ry0 + rh, rx0:rx0 + rw]
        src = sprite[ry0:ry0 + rh, rx0:rx0 + rw].astype(np.float32)
        roi = roi * (1.0 - 0.42 * sd)          # sombra no video
        roi = roi * (1.0 - a) + src * a        # Pikachu por cima
        frame[fy0:fy1, fx0:fx1] = np.clip(roi, 0, 255).astype(np.uint8)

        return frame

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
        elif self._last_rejection:
            cv2.putText(
                frame,
                f"ignorado: {self._last_rejection}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
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
        contour, metrics, rejection = self._select_hand(contours, faces, roi_x, roi_y)
        self._last_rejection = rejection

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

        if now < self.pikachu_until:
            shoulder = (int(frame.shape[1] * 0.82), int(frame.shape[0] * 0.72))
            elapsed = self.PIKACHU_SECONDS - (self.pikachu_until - now)
            self._draw_pikachu(frame, shoulder=shoulder, elapsed=elapsed)

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
