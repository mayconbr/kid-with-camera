"""Testes de classificacao (aberta/fechada) e da maquina de estados.

Rode com:  .venv/bin/python tests/test_gesture.py
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app import GestureDisappear

app = GestureDisappear.__new__(GestureDisappear)  # pula __init__ (sem camera)
app.gesture_state = app.STATE_NO_HAND
app.last_hand_time = 0.0
app._history = deque(maxlen=app.HISTORY_FRAMES)
app._candidate = None
app._candidate_votes = 0


def metrics_for(draw):
    img = np.zeros((720, 1280), np.uint8)
    draw(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    return app._hand_metrics(largest)


def fist(img):
    cv2.circle(img, (640, 360), 120, 255, -1)


def fist_thumb_out(img):
    cv2.circle(img, (660, 360), 110, 255, -1)
    cv2.ellipse(img, (545, 360), (55, 38), 0, 0, 360, 255, -1)


def fist_loose(img):
    cv2.ellipse(img, (640, 380), (125, 105), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (640, 300), (125, 45), 0, 0, 360, 255, -1)


def open_hand(img):
    cv2.ellipse(img, (640, 470), (105, 110), 0, 0, 360, 255, -1)
    fingers = [
        ((560, 340), 14, 75, 10),    # indicador
        ((615, 325), 15, 85, 0),     # medio
        ((668, 335), 14, 78, -8),    # anular
        ((715, 365), 12, 60, -18),   # minimo
        ((545, 470), 13, 52, -55),   # polegar
    ]
    for center, ax, ay, ang in fingers:
        cv2.ellipse(img, center, (ax, ay), ang, 0, 360, 255, -1)


def open_together(img):
    # mao aberta com dedos quase juntos. As pontas ficam em alturas
    # diferentes (medio > anular ~ indicador > minimo), como numa mao real:
    # pontas perfeitamente alinhadas fazem o casco convexo "pular" os dedos
    # do meio e fundir os vais num defeito so.
    cv2.ellipse(img, (640, 470), (105, 110), 0, 0, 360, 255, -1)
    for x, cy, ry in ((575, 338, 82), (610, 325, 90), (645, 330, 86), (680, 348, 70)):
        cv2.ellipse(img, (x, cy), (16, ry), 0, 0, 360, 255, -1)


def fist_bent_arm(img):
    """Punho ligado a um braco dobrado (caso real do usuario)."""
    cv2.circle(img, (640, 300), 110, 255, -1)                 # punho
    cv2.ellipse(img, (640, 520), (60, 190), 0, 0, 360, 255, -1)  # antibraco
    cv2.ellipse(img, (770, 640), (150, 55), 15, 0, 360, 255, -1)  # braco dobrado


# ---- 1. classificacao ----
cases = [
    ("punho               ", fist, False),
    ("punho c/ polegar    ", fist_thumb_out, False),
    ("punho relaxado      ", fist_loose, False),
    ("mao aberta          ", open_hand, True),
    ("mao aberta apertada ", open_together, True),
    ("punho + braco dobrad", fist_bent_arm, False),
]
for name, draw, expected_open in cases:
    m = metrics_for(draw)
    assert m is not None, f"{name}: contorno pequeno demais"
    status = "ABERTA" if m["is_open"] else "FECHADA"
    expected = "ABERTA" if expected_open else "FECHADA"
    print(f"{name}: area={int(m['area'])}  solidez={m['solidity']:.2f}  vales={m['deep_defects']}  -> {status} (esperado {expected})")
    assert m["is_open"] == expected_open, f"{name}: esperava {expected}, veio {status}"

print("regressao do braco dobrado: cavidades largas (axila/cotovelo) nao contam como vais de dedo")

# ---- 2. selecao: braco dobrado na zona do rosto NAO pode virar a "mao" ----
# Caso real relatado pelo usuario: um blob gigante de braco/tronco (com
# cavidades profundas mas LARGAS) cobria a zona do rosto e era selecionado
# como mao aberta, em vez do punho de verdade. O braco deve ser rejeitado
# como corpo e o punho selecionado.
select_img = np.zeros((720, 1280), np.uint8)
fist_bent_arm(select_img)                        # braco dobrado no meio do quadro
cv2.circle(select_img, (300, 550), 90, 255, -1)   # punho de verdade, a esquerda
contours, _ = cv2.findContours(select_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
assert len(contours) == 2

face_box = (640, 80, 200, 200)  # zona cabeca/pescoco cobre o braco, nao o punho
contour, metrics, rejection = app._select_hand(contours, [face_box], 0, 0)
assert contour is not None, f"punho deveria ser selecionado; rejeicao={rejection}"
bx, by, bw, bh = metrics["bbox"]
assert bx <= 300 <= bx + bw and by <= 550 <= by + bh, (
    f"o punho deveria ser a mao selecionada, veio bbox={metrics['bbox']}"
)
assert metrics["is_open"] is False, "punho selecionado deveria ser FECHADO"
assert rejection is None
print("regressao da selecao: braco dobrado na zona do rosto e ignorado; punho e a mao")

# ---- 3. maquina de estados ----
clock = [1000.0]


def tick():
    clock[0] += 0.05  # ~20 fps
    return clock[0]


def run_frames(metrics, n):
    for _ in range(n):
        app._update_state(metrics, tick())


def reset_state(state):
    app.gesture_state = state
    app.last_hand_time = 0.0
    app._history.clear()
    app._candidate = None
    app._candidate_votes = 0


m_fist = metrics_for(fist)
m_open = metrics_for(open_hand)

# sem mao -> mao aberta
reset_state(app.STATE_NO_HAND)
run_frames(None, 30)
assert app.gesture_state == app.STATE_NO_HAND, "deveria estar SEM MAO"
run_frames(m_open, 10)
assert app.gesture_state == app.STATE_OPEN, f"esperava ABERTA, veio {app.gesture_state}"

# REGRESSAO: bug relatado pelo usuario - frames ruidosos intercalados
# ("confirmando MAO FECHADA (x/4)" aparecia mas o status nunca mudava).
# Com a janela deslizante, um frame isolado classificado como o estado
# atual NAO pode zerar a contagem de votos.
reset_state(app.STATE_OPEN)
noisy = [m_fist, m_fist, m_open, m_fist, m_fist, m_open, m_fist, m_fist, m_open, m_fist]
for m in noisy:
    app._update_state(m, tick())
    if app.gesture_state == app.STATE_CLOSED:
        break
assert app.gesture_state == app.STATE_CLOSED, (
    f"ruido intercalado nao deveria travar a mudanca: estado={app.gesture_state}"
)
print("regressao do ruido: mudou para FECHADA mesmo com frames ruidosos intercalados")

# fechada -> aberta de novo
run_frames(m_open, 10)
assert app.gesture_state == app.STATE_OPEN, "deveria voltar para ABERTA"

# um frame isolado ruidoso nao pode trocar o estado sozinho
reset_state(app.STATE_OPEN)
app._update_state(m_fist, tick())
assert app.gesture_state == app.STATE_OPEN, "um unico frame de punho nao deveria trocar o estado"

# mao aparece ja fechada -> FECHADA direto (sem passar por ABERTA)
reset_state(app.STATE_NO_HAND)
run_frames(m_fist, 10)
assert app.gesture_state == app.STATE_CLOSED, "mao fechada deveria dar FECHADA direto"

# mao some por mais que o timeout -> SEM MAO
run_frames(None, 1)
assert app.gesture_state == app.STATE_CLOSED, "dropout curto nao deveria zerar o estado"
run_frames(None, 30)
assert app.gesture_state == app.STATE_NO_HAND, "sem mao por muito tempo deveria zerar"

print("\nOK: todos os testes passaram")
