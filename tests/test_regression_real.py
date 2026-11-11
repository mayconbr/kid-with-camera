"""Regressao com o caso real que motivou a correcao dos vais LARGOS.

O quadro `fixtures/recovered_frame.png` foi recuperado de uma captura de tela
do usuario (15/11/2026): punho fechado a esquerda, braco dobrado + tronco
formando um blob gigante sobre a zona do rosto. Na epoca o app selecionava o
blob do braco (solidez 0.41, 6 cavidades profundas) e mostrava MAO ABERTA
com a mao fechada. Hoje o braco deve ser rejeitado como corpo e o punho
(solidez ~0.94, sem vais) selecionado -> MAO FECHADA.

Rode com:  .venv/bin/python tests/test_regression_real.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app import GestureDisappear

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "recovered_frame.png")

app = GestureDisappear.__new__(GestureDisappear)
app.face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
)
app.profile_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
)
app._reset_tracking()


def load_clean_frame():
    """Frame do video do usuario com a interface do app apagada."""
    frame = cv2.imread(FIXTURE)
    assert frame is not None, f"fixture nao encontrada: {FIXTURE}"
    frame = frame.copy()
    frame[0:115, 0:720] = 0        # status + metricas + "confirmando"
    frame[535:720, 0:475] = 0      # painel da legenda
    for x in (102, 1177):          # bordas verticais da ROI
        frame[:, max(0, x - 6):x + 6] = 0
    for y in (14, 691):            # bordas horizontais da ROI
        frame[max(0, y - 6):y + 6, :] = 0
    return frame


# ---- 1. pipeline: o punho deve ser a mao selecionada, nao o braco ----
frame = load_clean_frame()

faces = app._detect_faces(frame)
assert faces, "o rosto do usuario deveria ser detectado no frame"
print(f"rosto detectado: {faces}")

roi_x, roi_y, roi_w, roi_h = app._roi(frame)
roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
mask = app._skin_mask(roi)
mask = app._remove_faces_from_mask(mask, faces, roi_x, roi_y)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

big = max(contours, key=cv2.contourArea)
m_big = app._hand_metrics(big)
print(f"maior blob: area={int(m_big['area'])} solidez={m_big['solidity']:.2f} "
      f"vais={m_big['deep_defects']} aberta={m_big['is_open']}")
# o maior blob e o braco/tronco: nao pode ser classificado como mao aberta
assert not m_big["is_open"], "blob do braco/tronco voltou a ser tratado como mao aberta (regressao)"

contour, metrics, rejection = app._select_hand(contours, faces, roi_x, roi_y)
assert contour is not None, f"punho deveria ser selecionado; rejeicao={rejection}"
bx, by, bw, bh = metrics["bbox"]
cx = bx + bw / 2 + roi_x
cy = by + bh / 2 + roi_y
print(f"mao selecionada: bbox_frame=({bx + roi_x},{by + roi_y},{bw},{bh}) "
      f"centro=({int(cx)},{int(cy)}) solidez={metrics['solidity']:.2f} vais={metrics['deep_defects']}")
assert metrics["is_open"] is False, "punho real do usuario deveria dar FECHADA"
assert metrics["solidity"] > 0.85, f"punho e liso (solidez alta); veio {metrics['solidity']:.2f}"
assert metrics["area"] < 60000, (
    f"deveria selecionar o punho (~25k px), veio area={int(metrics['area'])}"
)
print("punho real selecionado -> MAO FECHADA (braco/tronco rejeitado como corpo)")

# ---- 2. estado final apos varios frames ----
app._reset_tracking()
for i in range(12):
    processed = app.process_frame(load_clean_frame())
print(f"estado final: {app.gesture_state}")
assert app.gesture_state == app.STATE_CLOSED, (
    f"o frame real do usuario deveria estabilizar em MAO FECHADA; veio {app.gesture_state}"
)

dbg = load_clean_frame()
if contour is not None:
    cv2.drawContours(dbg, [contour], -1, (0, 255, 255), 2, offset=(roi_x, roi_y))
    cv2.rectangle(dbg, (bx + roi_x, by + roi_y), (bx + roi_x + bw, by + roi_y + bh), (255, 0, 0), 2)
for (fx, fy, fw, fh) in faces:
    cv2.rectangle(dbg, (fx, fy), (fx + fw, fy + fh), (255, 0, 255), 2)
out = os.path.join(HERE, "out", "recovered_analysis.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, dbg)

print("\nOK: regressao do caso real passou (punho -> MAO FECHADA)")
