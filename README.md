# audio-splitter
Esse projeto consiste em um separador de áudio que cria segmentos baseado em tempo de silêncio, ruído e falantes. O recurso de diarização é experimental e pode apresentar falhas.

*Requisitos:*
- 1 torradeira com placa de vídeo.
- ffmpeg
- python >= 3.9

> **Nota:** O `ffmpeg` é uma ferramenta externa. Certifique-se de instalá-lo no seu sistema (por exemplo, via `apt-get install ffmpeg` no Ubuntu ou utilizando outro gerenciador de pacotes adequado).

## Instalação

1. **Clone o repositório:**

  ```bash
   git clone https://github.com/birenes/audio-splitter
   cd audio-splitter
```   
2. **Instale os requisitos:**

  ```bash
  pip install -r requirements.txt
```
3. **Inicie o programa:**

```bash
python split_audio.py
