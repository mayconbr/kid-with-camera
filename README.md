# Camera Gesture Disappear

Projeto experimental para detectar uma mão abrindo e fechando na frente da webcam e mostrar um feedback visual curto na tela.

## Objetivo

A aplicação usa a webcam para acompanhar a mão dentro da região de interesse e mantém um status visível na tela:

- `SEM MAO` — nenhuma mão detectada
- `MAO ABERTA` — mão com dedos abertos/afastados
- `MAO FECHADA` — punho fechado

Quando a mão passa de **aberta** para **fechada** (gesto de abrir e fechar), a tela mostra um "OK" por alguns instantes.

## Como funciona

- captura a câmera com OpenCV
- cria uma máscara de pele (HSV) dentro da ROI e pega o maior contorno
- classifica a mão como aberta ou fechada usando duas métricas:
  - **solidez**: área do contorno dividida pela área do casco convexo. Punho fica próximo de `1.0`; mão aberta com dedos afastados fica bem abaixo.
  - **vales entre os dedos** (convexity defects): quantos espaços profundos existem entre os dedos. Mão aberta tem vários; punho praticamente nenhum.
- uma máquina de estados com anti-tremido (`MIN_CONFIRM_FRAMES`) só troca o status depois de alguns frames consecutivos com a mesma classificação, evitando piscadas de estado
- a transição `aberta -> fechada` dispara o feedback "OK"
- a janela da webcam fica aberta até pressionar `q`

## Legenda das cores na tela

| Cor | Elemento | Significado |
| --- | --- | --- |
| vermelho | retângulo externo | ROI: região da imagem onde a mão é procurada |
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
- coloque a mão dentro do retângulo vermelho (ROI), de preferência sem mais nada da cor de pele no quadro (rosto, braço etc. entram na detecção)
- abra a mão **afastando os dedos** — o status muda para `MAO ABERTA`
- feche a mão (punho) — o status muda para `MAO FECHADA` e o "OK" aparece
- pressione `q` para sair

## Ajustes finos

Os parâmetros ficam no topo da classe `GestureDisappear`, em `app.py`:

| Parâmetro | Padrão | O que faz |
| --- | --- | --- |
| `MIN_HAND_AREA` | `1800` | área mínima do contorno para considerar que há uma mão |
| `MIN_CONFIRM_FRAMES` | `4` | frames consecutivos para confirmar uma mudança de estado |
| `NO_HAND_TIMEOUT` | `0.4` | segundos sem mão para voltar para `SEM MAO` |
| `FEEDBACK_DURATION` | `0.8` | segundos que o "OK" fica na tela |
| `OPEN_MIN_DEFECTS` | `2` | vales profundos entre os dedos para considerar a mão aberta |
| `DEFECT_DEPTH_RATIO` | `0.08` | profundidade mínima de um vale (fração do tamanho da mão) |
| `OPEN_MAX_SOLIDITY` | `0.78` | solidez abaixo disso indica mão aberta (punho fica acima de `0.85`) |

Se o status estiver errado na sua iluminação/pele, observe as métricas no topo da tela e ajuste `OPEN_MAX_SOLIDITY` / `OPEN_MIN_DEFECTS` conforme os valores que aparecerem.

## Observações importantes

Para evitar o problema de incompatibilidade do MediaPipe no macOS, esta versão usa apenas OpenCV puro. Isso torna o projeto mais estável e mais fácil de rodar em computadores com câmera integrada.

Mão aberta com dedos **juntos demais** pode ser confundida com punho (as duas formas ficam parecidas para um contorno). Abra a mão afastando os dedos para o status `MAO ABERTA` funcionar bem.

Se a câmera não abrir, confira a permissão da webcam no macOS e no próprio terminal/VS Code.

## Arquivos

- `app.py` — lógica principal
- `requirements.txt` — dependências do projeto
