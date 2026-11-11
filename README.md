# Camera Gesture Disappear

Projeto experimental para detectar uma mão aberta ou fechada na frente da webcam e mostrar o status na tela.

## Objetivo

A aplicação usa a webcam para acompanhar a mão dentro da região de interesse e mantém um status visível na tela:

- `MAO NAO DETECTADA` — nenhuma mão detectada
- `MAO ABERTA` — mão com dedos abertos/afastados
- `MAO FECHADA` — punho fechado

Rostos são detectados (Haar cascade do OpenCV), destacados em **magenta** com a label `ROSTO` e **ignorados** pela detecção de mão — junto com a região da cabeça, pescoço e ombros — para que rosto e corpo não sejam confundidos com a mão.

## Como funciona

- captura a câmera com OpenCV
- detecta rostos de frente e de perfil (Haar cascades em imagem reduzida, ~12 ms/frame)
- cria uma máscara de pele (HSV) dentro da ROI e remove dela as regiões de rosto
- escolhe como mão o maior contorno de pele que parecer uma mão:
  - área mínima (`MIN_HAND_AREA`) e proporção largura/altura compatível com uma mão (blobs muito alongados, como braço/pescoço, são descartados)
  - **não pode estar colado na região do rosto/cabeça** (`HEAD_ZONE_*`): pescoço, ombros, peito e cabelo ficam nessa zona proibida
- classifica a mão como aberta ou fechada usando duas métricas:
  - **solidez**: área do contorno dividida pela área do casco convexo. Punho fica próximo de `1.0`; mão aberta com dedos afastados fica bem abaixo.
  - **vales entre os dedos** (convexity defects): quantos espaços profundos existem entre os dedos. Mão aberta tem vários; punho praticamente nenhum.
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

No topo da tela aparecem o status atual, as métricas ao vivo (`area`, `solidez`, `vales`) e, quando aplicável, o estado que está sendo confirmado. No canto inferior esquerdo fica a legenda das cores.

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
- coloque a mão dentro do retângulo vermelho (ROI), **afastada do rosto** (o rosto aparece numa caixa magenta e é ignorado)
- abra a mão **afastando os dedos** — o status muda para `MAO ABERTA`
- feche a mão (punho) — o status muda para `MAO FECHADA`
- tire a mão do quadro — o status muda para `MAO NAO DETECTADA`
- pressione `q` para sair

## Ajustes finos

Os parâmetros ficam no topo da classe `GestureDisappear`, em `app.py`:

| Parâmetro | Padrão | O que faz |
| --- | --- | --- |
| `MIN_HAND_AREA` | `1800` | área mínima do contorno para considerar que há uma mão |
| `HAND_MIN_ASPECT` / `HAND_MAX_ASPECT` | `0.25` / `4.0` | proporção largura/altura aceita para uma mão (fora disso: braço/pescoço) |
| `HISTORY_FRAMES` | `8` | tamanho da janela deslizante de classificações recentes |
| `CONFIRM_VOTES` | `5` | votos na janela necessários para confirmar uma mudança de estado |
| `NO_HAND_TIMEOUT` | `0.4` | segundos sem mão para voltar para `MAO NAO DETECTADA` |
| `OPEN_MIN_DEFECTS` | `2` | vales profundos entre os dedos para considerar a mão aberta |
| `DEFECT_DEPTH_RATIO` | `0.08` | profundidade mínima de um vale (fração do tamanho da mão) |
| `OPEN_MAX_SOLIDITY` | `0.78` | solidez abaixo disso indica mão aberta (punho fica acima de `0.85`) |
| `FACE_MIN_NEIGHBORS` | `5` | rigidez do detector de rosto (menor = detecta mais, com mais falsos positivos) |
| `HEAD_ZONE_SIDE` / `HEAD_ZONE_TOP` / `HEAD_ZONE_BOTTOM` | `0.8` / `0.8` / `1.2` | tamanho da zona proibida ao redor do rosto, em frações do tamanho do rosto. Se a sua mão estiver sendo ignorada perto do rosto, diminua; se o pescoço/ombros estiverem virando "mão", aumente |

Se o status estiver errado na sua iluminação/pele, observe as métricas no topo da tela e ajuste `OPEN_MAX_SOLIDITY` / `OPEN_MIN_DEFECTS` conforme os valores que aparecerem.

## Observações importantes

Para evitar o problema de incompatibilidade do MediaPipe no macOS, esta versão usa apenas OpenCV puro. Isso torna o projeto mais estável e mais fácil de rodar em computadores com câmera integrada.

Mão aberta com dedos **juntos demais** pode ser confundida com punho (as duas formas ficam parecidas para um contorno). Abra a mão afastando os dedos para o status `MAO ABERTA` funcionar bem.

A mão precisa estar **separada do rosto** no quadro: contornos de pele colados na região da cabeça (pescoço, ombros, cabelo) são tratados como corpo e ignorados. Se o seu rosto não estiver sendo detectado (caixa magenta não aparece), confira a iluminação — o detector funciona melhor com o rosto bem iluminado e de frente.

Se a câmera não abrir, confira a permissão da webcam no macOS e no próprio terminal/VS Code.

## Arquivos

- `app.py` — lógica principal
- `requirements.txt` — dependências do projeto
