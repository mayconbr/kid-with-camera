"""Verificacao visual por pixels: textos, caixas e legendas desenhados.

Rode com:  .venv/bin/python tests/smoke_visual.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app import GestureDisappear

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def make_app():
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


def draw_open_hand(img, cx=400, cy=380):
    cv2.ellipse(img, (cx, cy + 90), (105, 110), 0, 0, 360, SKIN, -1)
    fingers = [
        ((cx - 80, cy - 40), 14, 75, 10),
        ((cx - 25, cy - 55), 15, 85, 0),
        ((cx + 28, cy - 45), 14, 78, -8),
        ((cx + 75, cy - 15), 12, 60, -18),
        ((cx - 95, cy + 90), 13, 52, -55),
    ]
    for center, ax, ay, ang in fingers:
        cv2.ellipse(img, center, (ax, ay), ang, 0, 360, SKIN, -1)


def frame_with_face(hand=False):
    frame = np.full((720, 1280, 3), BACKGROUND, np.uint8)
    # rosto no lado direito (apos o flip do process_frame fica onde a mao nao esta)
    frame[120:120 + 360, 850:850 + 360] = cv2.resize(lena, (360, 360))
    if hand:
        draw_open_hand(frame, cx=350)
    return frame


def run_and_save(frame_fn, n, filename):
    for i in range(n):
        processed = app.process_frame(frame_fn())
    cv2.imwrite(f"{OUT}/{filename}", processed)
    print(f"salvo {filename}: estado={app.gesture_state}")
    return processed


frame_face_only = run_and_save(lambda: frame_with_face(False), 10, "v1_somente_rosto.png")
app._reset_tracking()
frame_hand = run_and_save(lambda: frame_with_face(True), 12, "v2_rosto_e_mao.png")


def count(img, color, region=None, tol=10):
    part = img if region is None else img[region[1]:region[3], region[0]:region[2]]
    diff = np.abs(part.astype(int) - np.array(color, int))
    return int(np.sum(np.all(diff <= tol, axis=2)))


def has(img, color, region, minimum, label):
    n = count(img, color, region)
    print(("OK " if n >= minimum else "FALHOU ") + label + f": {n} px")
    return n >= minimum


ok = True

# frame so com rosto (desenhado a direita; apos o flip do process_frame fica a esquerda)
ok &= has(frame_face_only, (160, 160, 160), (10, 10, 620, 60), 400, "texto MAO NAO DETECTADA (cinza)")
ok &= has(frame_face_only, (200, 200, 200), (10, 52, 720, 80), 25, "linha 'ignorado: ...' (cinza claro)")
ok &= has(frame_face_only, (255, 0, 255), (60, 100, 500, 700), 300, "caixa magenta do rosto")
ok &= has(frame_face_only, (255, 0, 255), (0, 540, 500, 720), 40, "legenda: linha magenta (rosto)")
ok &= has(frame_face_only, (0, 0, 255), (0, 540, 500, 720), 40, "legenda: linha vermelha")
ok &= has(frame_face_only, (0, 255, 255), (0, 540, 500, 720), 40, "legenda: linha amarela")
ok &= has(frame_face_only, (255, 0, 0), (0, 540, 500, 720), 40, "legenda: linha azul")
ok &= has(frame_face_only, (0, 255, 0), (0, 540, 500, 720), 40, "legenda: linha verde")

# frame com rosto + mao aberta (mao desenhada a esquerda; apos o flip fica a direita)
ok &= has(frame_hand, (0, 255, 0), (10, 10, 500, 60), 200, "texto MAO ABERTA (verde)")
ok &= has(frame_hand, (0, 255, 255), (750, 100, 1120, 640), 200, "contorno amarelo da mao")
ok &= has(frame_hand, (255, 0, 0), (750, 100, 1120, 640), 200, "caixa azul da mao")
ok &= has(frame_hand, (255, 0, 255), (60, 100, 500, 700), 300, "caixa magenta do rosto")

assert ok, "alguma verificacao visual falhou"
print("\nOK: teste visual passou (rosto destacado, mao detectada, legenda atualizada)")
