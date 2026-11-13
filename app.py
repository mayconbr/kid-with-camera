import math
import os
import time
from collections import deque

import cv2
import numpy as np


def _load_cascade(filename):
    """Carrega um classificador Haar que acompanha o opencv-python."""
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, filename))


_ELLIPSOID_CACHE = {}


def _ellipsoid_mesh(rx, ry, rz, nu=12, nv=9):
    """Malha de um elipsoide centrado na origem: (verts Nx3, faces Mx3)."""
    key = (round(rx, 3), round(ry, 3), round(rz, 3), nu, nv)
    cached = _ELLIPSOID_CACHE.get(key)
    if cached is not None:
        return cached
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nv + 1)
    cu, su = np.cos(u), np.sin(u)
    cvv, svv = np.cos(v), np.sin(v)
    verts = np.empty((nv + 1, nu, 3), np.float64)
    verts[..., 0] = (rx * cvv)[:, None] * cu[None, :]
    verts[..., 1] = (ry * svv)[:, None]
    verts[..., 2] = (rz * cvv)[:, None] * su[None, :]
    verts = verts.reshape(-1, 3)
    faces = []
    for i in range(nv):
        for j in range(nu):
            j2 = (j + 1) % nu
            a = i * nu + j
            b = i * nu + j2
            c = (i + 1) * nu + j2
            d = (i + 1) * nu + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.array(faces, np.int32)
    _ELLIPSOID_CACHE[key] = (verts, faces)
    return verts, faces


def _frustum_mesh(p0, p1, r0, r1, sides=12):
    """Tronco de cone fechado entre p0 e p1 (malha 3D)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    w = axis / length
    helper = np.array([0.0, 1.0, 0.0]) if abs(w[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(w, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w, e1)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    verts = np.vstack([p0 + r0 * ring, p1 + r1 * ring])
    faces = []
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((j, j2, sides + j2))
        faces.append((j, sides + j2, sides + j))
    for j in range(1, sides - 1):
        faces.append((0, j + 1, j))
        faces.append((sides, sides + j, sides + j + 1))
    return verts, np.array(faces, np.int32)


def _box_mesh(p0, p1, half_w, half_t):
    """Prisma (caixa) de p0 a p1, com largura no plano xy e espessura em z."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    d = d / length
    side = np.array([-d[1], d[0], 0.0])
    norm = float(np.linalg.norm(side))
    side = side / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])
    corners = [(-half_w, -half_t), (half_w, -half_t),
               (half_w, half_t), (-half_w, half_t)]
    verts = [p0 + a * side + b * zaxis for a, b in corners]
    verts += [p1 + a * side + b * zaxis for a, b in corners]
    verts = np.array(verts, float)
    faces = []
    for j in range(4):
        j2 = (j + 1) % 4
        faces.append((j, j2, 4 + j2))
        faces.append((j, 4 + j2, 4 + j))
    faces += [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    return verts, np.array(faces, np.int32)


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
        """Desenha um Pikachu 3D (malha) no ombro.

        O personagem e montado com malhas 3D (elipsoides, troncos de cone e
        prismas), projetado em perspectiva e sombreado por face em bandas
        (cel shading), com contorno escuro e sombra projetada. Quando
        `elapsed` (segundos desde o aparecimento) e informado, ele entra com
        um "pop", balanca, gira levemente, mexe orelhas/cauda e pisca.
        `elapsed=None` desenha a pose parada (util nos testes).
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if shoulder is None:
            shoulder = (int(w * 0.78), int(h * 0.72))
        sxp, syp = shoulder
        s = float(scale)

        # ---- animacao ----
        if elapsed is None:
            pop, bob, ear_sway, tail_sway, yaw = 1.0, 0.0, 0.0, 0.0, 0.0
            blink = False
        else:
            e = max(0.0, float(elapsed))
            t = min(1.0, e / 0.45)
            c1, c3 = 1.70158, 2.70158
            tm = t - 1.0
            pop = max(0.0, 1.0 + c3 * tm ** 3 + c1 * tm ** 2)  # ease-out-back
            bob = -abs(math.sin(e * 6.2)) * 6.0                # px
            ear_sway = math.sin(e * 5.0) * 6.0                 # graus
            tail_sway = math.sin(e * 3.0) * 12.0               # graus
            yaw = math.sin(e * 1.6) * 9.0                      # graus (giro 3D)
            blink = (e % 1.15) > 1.06

        # ---- paleta, luz e camera ----
        OUTLINE = (40, 64, 96)
        BLACK = (26, 26, 26)
        WHITE = (255, 255, 255)
        BASE = {
            "yellow": (48, 208, 248),
            "cheek": (40, 70, 240),
            "black": BLACK,
            "brown": (45, 100, 165),
        }
        BANDS = (0.58, 0.80, 1.00)          # multiplicadores do cel shading
        LIGHT = np.array([-0.42, -0.60, -0.68], float)
        LIGHT /= np.linalg.norm(LIGHT)      # luz de cima/esquerda/frente
        CAM = np.array([0.0, 0.0, -1000.0])  # camera em -z olhando para +z
        F = 1000.0
        PIVOT = np.array([0.0, 160.0, 0.0])  # pivo do "pop" (perto dos pes)

        ppu = s
        OX, OY = 300.0, 340.0                # deslocamento no sprite local
        WW = int(round(560.0 * ppu)) + 8
        HH = int(round(620.0 * ppu)) + 8
        canvas = np.zeros((HH, WW, 3), np.uint8)
        mask = np.zeros((HH, WW), np.uint8)   # silhueta (alpha) para a sombra

        cy, sy_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

        def rot_z(deg):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

        def to_world(p):
            """Giro global em y + pop em torno do pivo."""
            p = np.asarray(p, float)
            if yaw:
                x = p[..., 0] * cy + p[..., 2] * sy_
                z = -p[..., 0] * sy_ + p[..., 2] * cy
                p = np.stack([x, p[..., 1], z], axis=-1)
            return PIVOT + (p - PIVOT) * pop

        def project(p):
            p = to_world(p)
            den = np.maximum(F + p[..., 2], 1.0)
            u = (p[..., 0] * F / den + OX) * ppu
            v = (p[..., 1] * F / den + OY) * ppu + bob
            return u, v

        def render_part(verts, faces, center=(0.0, 0.0, 0.0), rot=None,
                        role="yellow", outline=True, pad=0.035, flat=None):
            """Projeta uma malha convexa, cula faces e pinta as bandas de luz."""
            v = np.asarray(verts, float)
            if rot is not None:
                v = v @ rot.T
            v = v + np.asarray(center, float)
            if faces.size == 0:
                return
            u, vv = project(v)
            pts = np.stack([u, vv], axis=-1)
            tri3 = v[faces]
            n = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
            ln = np.linalg.norm(n, axis=1)
            good = ln > 1e-6
            n[good] /= ln[good, None]
            fc = tri3.mean(axis=1)
            centroid = v.mean(axis=0)
            flip = np.sum(n * (fc - centroid), axis=1) < 0
            n[flip] *= -1.0                          # normais para fora
            facing = np.sum(n * (CAM - fc), axis=1) > 0
            use = good & facing
            if not np.any(use):
                return
            tri2d = pts[faces].astype(np.int32)
            hull = cv2.convexHull(pts.astype(np.int32))
            if len(hull) >= 3:
                hc = hull.reshape(-1, 2).mean(axis=0)
                grown = ((hull.reshape(-1, 2) - hc) * (1.0 + pad) + hc)
                grown = grown.astype(np.int32).reshape(-1, 1, 2)
                if flat is not None:
                    cv2.fillPoly(canvas, [grown], flat, cv2.LINE_AA)
                    cv2.fillPoly(mask, [grown], 255, cv2.LINE_AA)
                    return
                if outline:
                    cv2.fillPoly(canvas, [grown], OUTLINE, cv2.LINE_AA)
                cv2.fillPoly(mask, [grown], 255, cv2.LINE_AA)
                # fundo com a banda media: evita "costuras" escuras entre triangulos
                mid_col = tuple(int(ch * BANDS[1]) for ch in BASE[role])
                cv2.fillPoly(canvas, [hull.reshape(-1, 1, 2)], mid_col, cv2.LINE_AA)
            if flat is not None:
                return
            diff = np.clip(n @ LIGHT, 0.0, 1.0)
            bands = np.digitize(0.50 + 0.50 * diff, [0.70, 0.87])
            for k in range(3):
                sel = use & (bands == k)
                if not np.any(sel):
                    continue
                col = tuple(int(min(255, round(ch * BANDS[k])))
                            for ch in BASE[role])
                polys = [p.reshape(-1, 1, 2) for p in tri2d[sel]]
                cv2.fillPoly(canvas, polys, col, cv2.LINE_AA)

        def decal(center3, axes, color, start=0, end=360, thickness=-1):
            """Detalhe 2D (olho fechado, brilho, boca) projetado sobre a malha."""
            u, vv = project(np.asarray([center3], float))
            c = (int(round(u[0])), int(round(vv[0])))
            ax = (max(1, int(round(axes[0] * ppu))),
                  max(1, int(round(axes[1] * ppu))))
            th = thickness if thickness < 0 else max(1, int(round(thickness * ppu)))
            cv2.ellipse(canvas, c, ax, 0, start, end, color, th, cv2.LINE_AA)

        def decal_line(p0, p1, color, thickness):
            u0, v0 = project(np.asarray([p0], float))
            u1, v1 = project(np.asarray([p1], float))
            cv2.line(canvas, (int(round(u0[0])), int(round(v0[0]))),
                     (int(round(u1[0])), int(round(v1[0]))), color,
                     max(1, int(round(thickness * ppu))), cv2.LINE_AA)

        # ---- cauda de raio 3D (atras de tudo), girando em torno da base ----
        tail_base = np.array([84.0, 104.0, -40.0])
        ang = math.radians(tail_sway)
        tca, tsa = math.cos(ang), math.sin(ang)

        def tail_pt(p):
            d = np.asarray(p, float) - tail_base
            return tail_base + np.array([d[0] * tca - d[1] * tsa,
                                         d[0] * tsa + d[1] * tca, d[2]])

        p0 = tail_pt((84, 104, -40))
        p1 = tail_pt((112, 88, -40))
        p2 = tail_pt((138, 66, -40))
        p3 = tail_pt((104, 38, -40))
        p4 = tail_pt((154, 2, -40))
        tip = p4 + np.array([0.723, -0.690, 0.0]) * 40.0
        tail = [(p0, p1, 17.0, "brown"), (p1, p2, 13.0, "yellow"),
                (p2, p3, 13.0, "yellow"), (p3, p4, 13.0, "yellow"),
                (p4, tip, 30.0, "yellow")]
        # contorno continuo: uma silhueta escura por tras de todos os gomos
        for a_pt, b_pt, hw, _role in tail:
            vb, fb = _box_mesh(a_pt, b_pt, hw + 3.5, 11.5)
            render_part(vb, fb, flat=OUTLINE)
        for a_pt, b_pt, hw, role in tail:
            vb, fb = _box_mesh(a_pt, b_pt, hw, 8.0)
            render_part(vb, fb, role=role, outline=False)

        # ---- pes ----
        v_foot, f_foot = _ellipsoid_mesh(58, 30, 44, 11, 8)
        for fx in (-52.0, 52.0):
            render_part(v_foot, f_foot, center=(fx, 158.0, 26.0))

        # ---- corpo ----
        v_body, f_body = _ellipsoid_mesh(88, 82, 74, 15, 11)
        render_part(v_body, f_body, center=(0.0, 84.0, 0.0))

        # ---- orelhas (atras da cabeca), com ponta preta e balanco ----
        for sign in (-1.0, 1.0):
            base = np.array([sign * 38.0, -128.0, 0.0])
            tip_pt = np.array([sign * 122.0, -282.0, 8.0])
            a = math.radians(ear_sway * (-1.0 if sign < 0 else 1.0))
            ea, eb = math.cos(a), math.sin(a)
            d = tip_pt - base
            tip_pt = base + np.array([d[0] * ea - d[1] * eb,
                                      d[0] * eb + d[1] * ea, d[2]])
            mid = base + (tip_pt - base) * 0.76
            vb, fb = _frustum_mesh(base, mid, 28.0, 20.0)
            render_part(vb, fb, role="yellow")
            vb, fb = _frustum_mesh(mid, tip_pt, 20.0, 9.0)
            render_part(vb, fb, role="black")

        # ---- cabeca grande ----
        v_head, f_head = _ellipsoid_mesh(114, 96, 92, 18, 13)
        render_part(v_head, f_head, center=(0.0, -66.0, 0.0))

        # ---- bracos e maos ----
        v_arm, f_arm = _ellipsoid_mesh(38, 21, 21, 11, 8)
        v_hand, f_hand = _ellipsoid_mesh(17, 17, 17, 10, 8)
        for sign in (-1.0, 1.0):
            render_part(v_arm, f_arm, center=(sign * 88.0, 54.0, 12.0),
                        rot=rot_z(sign * 50.0))
            render_part(v_hand, f_hand, center=(sign * 115.0, 24.0, 24.0))

        # ---- bochechas vermelhas (sobre a superficie da cabeca) ----
        v_cheek, f_cheek = _ellipsoid_mesh(22, 22, 11, 10, 7)
        for cx in (-84.0, 84.0):
            render_part(v_cheek, f_cheek, center=(cx, -26.0, 55.0),
                        role="cheek", outline=False)

        # ---- olhos (3D) com brilho, ou arco fechado na piscada ----
        v_eye, f_eye = _ellipsoid_mesh(20, 25, 9, 10, 7)
        for eye_x in (-42.0, 42.0):
            if blink:
                decal((eye_x, -70.0, 92.0), (20, 16), BLACK, 20, 160, 4)
            else:
                render_part(v_eye, f_eye, center=(eye_x, -76.0, 86.0),
                            role="black", outline=False)
                decal((eye_x - 5.0, -85.0, 98.0), (8, 8), WHITE)
                decal((eye_x + 7.0, -66.0, 96.0), (3.6, 3.6), WHITE)

        # ---- nariz minusculo ----
        v_nose, f_nose = _ellipsoid_mesh(6, 4, 4, 6, 4)
        render_part(v_nose, f_nose, center=(0.0, -46.0, 92.0),
                    role="black", outline=False)

        # ---- sorriso em "w" ----
        for mouth_x in (-23.0, 23.0):
            decal((mouth_x, -24.0, 86.0), (24, 16), BLACK, 0, 180, 4)

        # ---- dedos dos pes ----
        for fx in (-52.0, 52.0):
            for toe in (-16.0, 0.0, 16.0):
                decal_line((fx + toe, 168.0, 66.0),
                           (fx + toe - 3.0, 186.0, 66.0), OUTLINE, 3)

        # ---- sombra projetada (sombra da silhueta, deslocada para baixo/direita) ----
        q = 4.0
        sqw, sqh = max(1, int(WW / q)), max(1, int(HH / q))
        small_mask = cv2.resize(mask, (sqw, sqh), interpolation=cv2.INTER_AREA)
        sh = cv2.GaussianBlur(small_mask.astype(np.float32), (0, 0),
                              max(1.0, 4.5 * ppu / q))
        shift = np.float32([[1, 0, 9.0 * s / q], [0, 1, 11.0 * s / q]])
        shadow = cv2.warpAffine(sh, shift, (sqw, sqh), flags=cv2.INTER_LINEAR,
                                borderValue=0)

        # ---- composicao sobre o video ----
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        shad = cv2.resize(shadow / 255.0, (WW, HH),
                          interpolation=cv2.INTER_LINEAR)[..., None]
        x0 = int(round(sxp - OX * s))
        y0 = int(round(syp - OY * s))
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, x0 + WW), min(h, y0 + HH)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        rx0, ry0 = fx0 - x0, fy0 - y0
        rw, rh = fx1 - fx0, fy1 - fy0

        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        a = alpha[ry0:ry0 + rh, rx0:rx0 + rw]
        sd = shad[ry0:ry0 + rh, rx0:rx0 + rw]
        src = canvas[ry0:ry0 + rh, rx0:rx0 + rw].astype(np.float32)
        roi = roi * (1.0 - 0.42 * sd)          # sombra projetada no video
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
            shoulder = (int(frame.shape[1] * 0.78), int(frame.shape[0] * 0.72))
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
