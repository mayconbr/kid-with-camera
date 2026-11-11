# Camera Gesture Disappear

Projeto experimental para detectar um movimento brusco da mão e mostrar um feedback visual curto na tela.

## Objetivo

A aplicação usa a webcam e processamento de diferença entre frames para identificar um movimento forte, parecido com um estalo na frente da câmera. Quando o gesto é reconhecido, a tela mostra um texto curto, como "OK", em vez de se tornar preta.

## Como funciona

- captura a câmera com OpenCV
- converte cada frame para cinza
- compara o frame atual com o anterior
- calcula a quantidade de movimento
- quando há um pico rápido, dispara o feedback visual
- a janela da webcam fica aberta até pressionar `q`

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
- mova a mão rapidamente na frente da câmera
- o gesto deve disparar um texto "OK" por um instante
- pressione `q` para sair

## Observações importantes

Para evitar o problema de incompatibilidade do MediaPipe no macOS, esta versão usa apenas OpenCV puro. Isso torna o projeto mais estável e mais fácil de rodar em computadores com câmera integrada.

Se a câmera não abrir, confira a permissão da webcam no macOS e no próprio terminal/VS Code.

## Arquivos

- `app.py` — lógica principal
- `requirements.txt` — dependências do projeto
