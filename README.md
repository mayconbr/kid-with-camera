# Camera Gesture Disappear

Projeto experimental para detectar uma mão aberta ou fechada na frente da webcam e mostrar o status na tela.

## Objetivo

A aplicação usa a webcam para acompanhar a mão dentro da região de interesse e mantém um status visível na tela:

- `MAO NAO DETECTADA` — nenhuma mão detectada
- `MAO ABERTA` — mão com dedos abertos/afastados
- `MAO FECHADA` — punho fechado

Rostos são detectados (Haar cascade do OpenCV), destacados em **magenta** com a label `ROSTO` e **ignorados** pela detecção de mão. A distinção entre mão e corpo é feita principalmente pelo **formato do contorno**: uma mão aberta (com vales entre os dedos) é aceita em qualquer lugar do quadro; um blob liso (punho, pescoço, ombro) só é aceito fora da região da cabeça/pescoço.

## Como funciona

- captura a câmera com OpenCV
- detecta rostos de frente e de perfil (Haar cascades em imagem reduzida, ~12 ms/frame)
- cria uma máscara de pele (HSV) dentro da ROI e remove dela a região do rosto (com margens pequenas, para não apagar uma mão ao lado do rosto)
- escolhe como mão o maior contorno de pele que parecer uma mão:
  - área mínima (`MIN_HAND_AREA`) e proporção largura/altura compatível com uma mão (blobs muito alongados, como braço/pescoço, são descartados)
  - **formato primeiro**: um contorno com `ZONE_BYPASS_DEFECTS` ou mais vales profundos entre os dedos é aceito **em qualquer lugar** do quadro, até colado no rosto — pescoço e ombros nunca têm esse formato
  - um contorno **liso** (sem vales, como um punho) só é aceito **fora da zona da cabeça/pescoço** (`HEAD_MARGIN_*` e `NECK_DEPTH`); dentro dessa zona ele é tratado como corpo, e a tela mostra o motivo (ex.: `ignorado: blob liso na zona do rosto/pescoco`)
- classifica a mão como aberta ou fechada contando os **vales entre os dedos** (convexity defects):
  - mão aberta tem vários; punho fechado praticamente nenhum
  - um vale só conta como "vale de dedo" se for **profundo** (`DEFECT_DEPTH_RATIO`) **e estreito** (`DEFECT_MAX_CHORD`, distância entre as bordas do vale). Braços dobrados e troncos também têm cavidades profundas — axila, cotovelo —, mas são cavidades **largas**, então não são confundidas com o espaço entre dois dedos
  - a **solidez** (área do contorno ÷ área do casco convexo) aparece na tela apenas como informação: um braço dobrado também tem solidez baixa, então ela não decide o status
- uma máquina de estados com janela deslizante confirma a mudança de status: o estado só troca quando `CONFIRM_VOTES` dos últimos `HISTORY_FRAMES` frames apontam para o mesmo estado novo. Assim, um frame isolado ruidoso não zera a contagem (o que travava o status em "confirmando")
- a janela da webcam fica aberta até pressionar `q`

## Legenda das cores na tela

| Cor | Elemento | Significado |
| --- | --- | --- |
| vermelho | retângulo externo | ROI: região da imagem onde a mão é procurada |
| magenta | caixa com a label `ROSTO` | rosto detectado (ignorado pela detecção de mão) |
| amarelo | contorno | contorno da mão detectada pela máscara de pele |
| azul | caixa | caixa delimitadora (bounding box) da mão |
| verde | eixos | eixos do centro da caixa da mão |

No topo da tela aparecem o status atual, as métricas ao vivo (`area`, `solidez`, `vales`), o estado que está sendo confirmado e, quando um blob de pele grande é descartado, o motivo (linha `ignorado: ...`). No canto inferior esquerdo fica a legenda das cores.

## Requisitos

- Python 3.9+
- webcam do computador
- permissão de câmera liberada para o VS Code ou Terminal no macOS

## Instalação

```bash
cd camera-gesture-disappear
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução

```bash
python app.py
```

## Como usar

- a janela da webcam deve abrir
- coloque a mão dentro do retângulo vermelho (ROI); a mão aberta funciona **em qualquer posição**, até perto do rosto
- abra a mão **afastando os dedos** — o status muda para `MAO ABERTA`
- feche a mão (punho) — o status muda para `MAO FECHADA` (mantenha o punho afastado do rosto/pescoço e do peito)
- tire a mão do quadro — o status muda para `MAO NAO DETECTADA`
- pressione `q` para sair

## Pikachu no ombro

Quando o status confirma `MAO FECHADA`, um Pikachu estilizado aparece no canto inferior direito por alguns segundos (`PIKACHU_SECONDS`). Ele é desenhado só com OpenCV (sem sprites externos) e tem:

- volume 3D: o Pikachu é uma malha 3D montada no próprio código (elipsoides do corpo/cabeça, troncos de cone das orelhas e prismas da cauda), projetada em perspectiva, com face culling, iluminação por face em bandas (cel shading), contorno escuro e sombra suave projetada no vídeo
- animação: entra com um "pop" (ease-out-back), balança o corpo, gira levemente em 3D, mexe as orelhas e a cauda e pisca de vez em quando
- visual do anime: amarelo vivo, orelhas com ponta preta, bochechas vermelhas, sorriso em "w" e cauda de raio

Não é usado nenhum arquivo de imagem/modelo externo: tudo é gerado em `app.py`. A posição padrão fica um pouco mais para a esquerda (78% da largura) para a cauda não encostar na borda da tela.

## Ajustes finos

Os parâmetros ficam no topo da classe `GestureDisappear`, em `app.py`:

| Parâmetro | Padrão | O que faz |
| --- | --- | --- |
| `MIN_HAND_AREA` | `1800` | área mínima do contorno para considerar que há uma mão |
| `HAND_MIN_ASPECT` / `HAND_MAX_ASPECT` | `0.25` / `4.0` | proporção largura/altura aceita para uma mão (fora disso: braço/pescoço) |
| `HISTORY_FRAMES` | `8` | tamanho da janela deslizante de classificações recentes |
| `CONFIRM_VOTES` | `5` | votos na janela necessários para confirmar uma mudança de estado |
| `NO_HAND_TIMEOUT` | `0.4` | segundos sem mão para voltar para `MAO NAO DETECTADA` |
| `OPEN_MIN_DEFECTS` | `2` | vales entre dedos (profundos e estreitos) para considerar a mão aberta |
| `DEFECT_DEPTH_RATIO` | `0.08` | profundidade mínima de um vale (fração do tamanho da mão) |
| `DEFECT_MAX_CHORD` | `0.22` | largura máxima de um vale entre dedos (fração do tamanho da mão). Cavidades mais largas (axila, cotovelo) são ignoradas — em uma mão real o espaço entre dois dedos é estreito |
| `FACE_MIN_NEIGHBORS` | `5` | rigidez do detector de rosto (menor = detecta mais, com mais falsos positivos) |
| `FACE_MASK_PAD_SIDE` / `FACE_MASK_PAD_TOP` / `FACE_MASK_PAD_BOTTOM` | `0.10` / `0.15` / `0.40` | margem usada para apagar o rosto da máscara de pele (fração do tamanho do rosto). Se uma mão colada no rosto estiver sendo apagada, diminua |
| `HEAD_MARGIN_SIDE` / `HEAD_MARGIN_TOP` | `0.25` / `0.35` | zona da cabeça ao redor do rosto (fração do tamanho do rosto). Punhos lisos dentro dessa zona são tratados como corpo |
| `NECK_DEPTH` | `0.9` | profundidade da zona abaixo do rosto (fração da altura do rosto), cobrindo pescoço. Se o seu punho na altura do peito estiver sendo ignorado, diminua; se o pescoço estiver virando "mão", aumente |
| `ZONE_BYPASS_DEFECTS` | `3` | vales profundos entre os dedos para aceitar uma mão aberta **dentro** da zona do rosto |
| `PIKACHU_SECONDS` | `2.6` | segundos que o Pikachu fica na tela depois de confirmar `MAO FECHADA` |

Se o status estiver errado na sua iluminação/pele, observe as métricas no topo da tela e ajuste `DEFECT_MAX_CHORD` / `OPEN_MIN_DEFECTS` conforme os valores que aparecerem (a solidez agora é só informativa).

## Observações importantes

Para evitar o problema de incompatibilidade do MediaPipe no macOS, esta versão usa apenas OpenCV puro. Isso torna o projeto mais estável e mais fácil de rodar em computadores com câmera integrada.

Mão aberta com os dedos **completamente juntos** forma um "remo" sem vales e pode ser confundida com punho (as duas formas ficam parecidas para um contorno). Na prática os dedos ficam levemente afastados e com pontas em alturas diferentes, o que basta para os vales aparecerem. Se quiser garantir o status `MAO ABERTA`, abra a mão afastando um pouco os dedos.

A **mão aberta funciona em qualquer posição** do quadro, inclusive perto do rosto — o formato com vales entre os dedos tem prioridade sobre a posição. Já o **punho fechado** precisa ficar afastado do rosto/pescoço e não colado ao peito: um blob liso nessas regiões é tratado como corpo (o motivo aparece na linha `ignorado: ...` no topo da tela). Se o seu rosto não estiver sendo detectado (caixa magenta não aparece), confira a iluminação — o detector funciona melhor com o rosto bem iluminado e de frente.

Se a câmera não abrir, confira a permissão da webcam no macOS e no próprio terminal/VS Code.

## Testes

A pasta `tests/` traz testes automatizados (sem precisar de câmera) que cobrem a classificação, a máquina de estados, a zona do rosto/pescoço, uma verificação visual por pixels e uma regressão com um frame real de webcam:

```bash
.venv/bin/python tests/test_gesture.py          # classificacao + maquina de estados + selecao mao vs braco + Pikachu 3D/animacao
.venv/bin/python tests/test_face.py             # rosto, zona cabeca/pescoco, bypass da mao aberta
.venv/bin/python tests/test_regression_real.py  # frame real recuperado de uma captura de tela (punho + braco dobrado)
.venv/bin/python tests/smoke_visual.py           # verifica por pixels os textos/caixas/legendas desenhados
```

Imagens de saída para inspeção ficam em `tests/out/` (fora do git).

## Arquivos

- `app.py` — lógica principal
- `requirements.txt` — dependências do projeto
- `tests/` — testes automatizados e fixtures (`tests/fixtures/`)
