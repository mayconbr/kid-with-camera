# Camera Gesture Disappear

Projeto simples para testar um efeito de "sumir da câmera" usando processamento de imagem em tempo real com OpenCV.

## Objetivo

Esse código foi criado como uma base experimental para detectar um gesto simples com a mão e trocar a imagem da câmera por uma tela invisível.

## Como funciona

- captura a webcam com OpenCV
- detecta regiões de pele na imagem
- verifica se houve movimento/gesto relevante
- quando o gesto dispara, ativa o efeito de invisibilidade
- a aplicação funciona em preview local; para usar em Google Meet/Teams, o passo seguinte é conectar esse fluxo a uma câmera virtual externa

## Requisitos

- Python 3.9+
- webcam do computador
- permissão de câmera liberada para o VS Code ou Terminal no macOS

## Instalação

```bash
cd camera-gesture-disappear
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python app.py
```

## Como usar

- a janela do OpenCV deve abrir com a câmera
- movimente a mão em frente à câmera para disparar o gesto
- pressione `q` para sair

## Observações importantes

Este projeto é uma base simples e funcional para testes locais. Ele não invade o Google Meet ou Teams diretamente.

Para usar no Meet/Teams, normalmente o caminho correto é:

1. rodar a aplicação localmente
2. enviar a saída para uma câmera virtual (por exemplo, OBS Virtual Camera)
3. selecionar essa câmera virtual dentro do app de reunião

Em macOS, o fator mais comum de erro é a permissão da câmera do sistema para o VS Code ou Terminal.

## Arquivos

- `app.py` — lógica principal
- `requirements.txt` — dependências
