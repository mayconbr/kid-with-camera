"""Testes de deteccao de rosto e da zona cabeca/pescoco.

Rode com:  .venv/bin/python tests/test_face.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app import GestureDisappear

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def make_app():
    """Instancia sem camera, com cascades reais."""
    app = GestureDisappear.__new__(GestureDisappear)
    app.face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    )
    app.profile_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
    )
    app._reset_tracking()
    return app


app = make_app()

SKIN = (90, 130, 190)      # BGR que passa na mascara de pele
BACKGROUND = (25, 25, 25)  # fundo escuro (rejeitado pela mascara)

lena = cv2.imread(os.path.join(HERE, "fixtures", "lena.jpg"))
assert lena is not None, "lena.jpg nao encontrada"


def draw_open_hand(img, cx=880, cy=380, s=1.0):
    cv2.ellipse(img, (cx, int(cy + 90 * s)), (int(105 * s), int(110 * s)), 0, 0, 360, SKIN, -1)
    fingers = [
        ((cx - 80, int(cy - 40 * s)), int(14 * s), int(75 * s), 10),
        ((cx - 25, int(cy - 55 * s)), int(15 * s), int(85 * s), 0),
        ((cx + 28, int(cy - 45 * s)), int(14 * s), int(78 * s), -8),
        ((cx + 75, int(cy - 15 * s)), int(12 * s), int(60 * s), -18),
        ((cx - 95, int(cy + 90 * s)), int(13 * s), int(52 * s), -55),
    ]
    for center, ax, ay, ang in fingers:
        cv2.ellipse(img, center, (ax, ay), ang, 0, 360, SKIN, -1)


def base_frame():
    return np.full((720, 1280, 3), BACKGROUND, np.uint8)


# ---- 1. o detector acha o rosto na imagem de teste ----
faces = app._detect_faces(lena)
print(f"rostos detectados na imagem de teste: {faces}")
assert faces, "nenhum rosto detectado na lena.jpg"


# ---- 2. so o rosto no quadro, sem mao -> MAO NAO DETECTADA ----
app._reset_tracking()


def run_seq(frames, n, save=None):
    last = None
    for i in range(n):
        frame = frames() if callable(frames) else frames
        last = app.process_frame(frame)
        if save and i == n - 1:
            cv2.imwrite(f"{OUT}/{save}", last)
    return last


def face_only():
    frame = base_frame()
    frame[120:120 + 360, 60:60 + 360] = cv2.resize(lena, (360, 360))
    return frame


# sem mao, com rosto: apos o timeout de "sem mao" o estado deve zerar
start = time.time()
for i in range(25):
    processed = app.process_frame(face_only())
    time.sleep(0.03)  # deixa o relogio andar (25 * 0.03 + processamento > timeout)
elapsed = time.time() - start
print(f"estado apos {elapsed:.1f}s so com o rosto no quadro: {app.gesture_state}")
cv2.imwrite(f"{OUT}/face_only.png", processed)
assert app.gesture_state == app.STATE_NO_HAND, (
    f"rosto nao deveria ser detectado como mao; estado={app.gesture_state}"
)
print("rosto no quadro sem mao -> MAO NAO DETECTADA")

# ---- 3. rosto + mao aberta -> mao detectada como ABERTA ----
app._reset_tracking()


def face_and_hand():
    frame = face_only()
    draw_open_hand(frame)
    return frame


for i in range(15):
    processed = app.process_frame(face_and_hand())
print(f"estado com rosto + mao aberta: {app.gesture_state}")
cv2.imwrite(f"{OUT}/face_and_hand.png", processed)
assert app.gesture_state == app.STATE_OPEN, (
    f"mao aberta deveria ser detectada mesmo com o rosto no quadro; estado={app.gesture_state}"
)
print("rosto + mao aberta -> MAO ABERTA (rosto ignorado)")


# ---- helpers para os testes da zona cabeca/pescoco ----
def flipped(frame):
    """O process_frame espelha o frame; os testes desenham no original."""
    return cv2.flip(frame, 1)


def zone_of(face, side_frac, top_frac, bottom_frac):
    fx, fy, fw, fh = face
    side = int(fw * side_frac)
    return (fx - side, fy - int(fh * top_frac), fx + fw + side, fy + fh + int(fh * bottom_frac))


def bbox_overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def fist_bbox_flipped(cx_original, cy, r):
    """bbox (x1, y1, x2, y2) de um punho desenhado no frame ORIGINAL,
    convertido para as coordenadas do frame espelhado."""
    return (1280 - cx_original - r, cy - r, 1280 - cx_original + r, cy + r)


# ---- 4. punho AO LADO do rosto: agora e aceito (zona menor + formato) ----
app._reset_tracking()

FIST_BESIDE = (497, 320)  # punho no frame ORIGINAL, ao lado do rosto
FIST_R = 65


def face_and_fist_beside():
    frame = face_only()
    cv2.circle(frame, FIST_BESIDE, FIST_R, SKIN, -1)
    return frame


# geometrico: o punho fica DENTRO da zona antiga (0.8) e FORA da nova (0.25)
faces_now = app._detect_faces(flipped(face_and_fist_beside()))
assert faces_now, "rosto deveria continuar detectado com o punho ao lado"
old_zone = zone_of(faces_now[0], 0.8, 0.8, 1.2)
new_zone = zone_of(faces_now[0], 0.25, 0.35, 0.9)
fist_box = fist_bbox_flipped(*FIST_BESIDE, FIST_R)
assert bbox_overlaps(fist_box, old_zone), "cenario deveria estar dentro da zona ANTIGA (regressao)"
assert not bbox_overlaps(fist_box, new_zone), "punho deveria estar fora da zona NOVA"

for i in range(10):
    processed = app.process_frame(face_and_fist_beside())
print(f"estado com punho ao lado do rosto: {app.gesture_state}")
cv2.imwrite(f"{OUT}/fist_beside_face.png", processed)
assert app.gesture_state == app.STATE_CLOSED, (
    f"punho ao lado do rosto deveria ser detectado; estado={app.gesture_state}"
)
assert app._last_rejection is None
print("punho ao lado do rosto -> MAO FECHADA (zona nova nao engole a mao)")

# ---- 5. punho DENTRO da zona cabeca/pescoco continua ignorado ----
app._reset_tracking()

FIST_IN_ZONE = (275, 557)  # original: apos o flip fica sob o queixo/pescoco


def face_and_fist_in_zone():
    frame = face_only()
    cv2.circle(frame, FIST_IN_ZONE, FIST_R, SKIN, -1)
    return frame


faces_now = app._detect_faces(flipped(face_and_fist_in_zone()))
assert faces_now, "rosto deveria continuar detectado"
fist_box = fist_bbox_flipped(*FIST_IN_ZONE, FIST_R)
new_zone = zone_of(faces_now[0], 0.25, 0.35, 0.9)
assert bbox_overlaps(fist_box, new_zone), "punho deveria estar dentro da zona nova"

for i in range(10):
    processed = app.process_frame(face_and_fist_in_zone())
print(f"estado com punho na zona do rosto: {app.gesture_state} (rejeicao: {app._last_rejection})")
cv2.imwrite(f"{OUT}/fist_in_zone.png", processed)
assert app.gesture_state == app.STATE_NO_HAND, (
    f"punho colado na cabeca/pescoco nao e mao; estado={app.gesture_state}"
)
assert app._last_rejection is not None, "deveria mostrar o motivo da rejeicao na tela"
print("punho na zona do rosto/pescoco -> ignorado (motivo exibido na tela)")

# ---- 6. punho na altura do peito (ABAIXO da zona) e aceito ----
app._reset_tracking()

FIST_CHEST = (280, 610)  # original: apos o flip fica abaixo do fim da zona


def face_and_fist_chest():
    frame = face_only()
    cv2.circle(frame, FIST_CHEST, FIST_R, SKIN, -1)
    return frame


faces_now = app._detect_faces(flipped(face_and_fist_chest()))
assert faces_now, "rosto deveria continuar detectado"
fist_box = fist_bbox_flipped(*FIST_CHEST, FIST_R)
new_zone = zone_of(faces_now[0], 0.25, 0.35, 0.9)
assert not bbox_overlaps(fist_box, new_zone), "punho deveria estar abaixo da zona"

for i in range(10):
    processed = app.process_frame(face_and_fist_chest())
print(f"estado com punho na altura do peito: {app.gesture_state}")
cv2.imwrite(f"{OUT}/fist_chest.png", processed)
assert app.gesture_state == app.STATE_CLOSED, (
    f"punho na altura do peito deveria ser detectado; estado={app.gesture_state}"
)
print("punho na altura do peito (abaixo da zona) -> MAO FECHADA")

# ---- 7. mao ABERTA dentro da zona e aceita (bypass pelo formato) ----
# teste direto do _select_hand com mao em tamanho cheio (sem recorte da ROI):
# a zona do rosto cobre a mao, mas os vais entre os dedos tem prioridade
zone_frame = np.zeros((720, 1280, 3), np.uint8)
draw_open_hand(zone_frame, cx=640, cy=360)
zone_mask = app._skin_mask(zone_frame)
zone_contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hand_metrics = app._hand_metrics(max(zone_contours, key=cv2.contourArea))
assert hand_metrics and hand_metrics["deep_defects"] >= app.ZONE_BYPASS_DEFECTS, (
    f"mao de teste precisa ter vais profundos; metricas={hand_metrics}"
)
face_box = (560, 200, 140, 140)  # zona cabeca/pescoco cobre a mao
contour, metrics, rejection = app._select_hand(zone_contours, [face_box], 0, 0)
assert contour is not None and metrics["is_open"], (
    f"mao aberta com vales deveria ser aceita dentro da zona; rejeicao={rejection}"
)
print("mao aberta dentro da zona -> aceita (bypass pelo formato dos dedos)")

# o mesmo lugar com um punho liso -> rejeitado
fist_frame = np.zeros((720, 1280, 3), np.uint8)
cv2.circle(fist_frame, (640, 360), 90, SKIN, -1)
fist_mask = app._skin_mask(fist_frame)
fist_contours, _ = cv2.findContours(fist_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour, metrics, rejection = app._select_hand(fist_contours, [face_box], 0, 0)
assert contour is None and rejection is not None, (
    "punho liso dentro da zona deveria ser rejeitado"
)
print("punho liso dentro da zona -> rejeitado (motivo exibido na tela)")

# ---- 8. performance da deteccao de rosto ----
frame = face_only()
t0 = time.time()
for _ in range(20):
    app._detect_faces(frame)
dt = (time.time() - t0) / 20 * 1000
print(f"deteccao de rosto: {dt:.1f} ms/frame")

print("\nOK: testes de rosto passaram")
