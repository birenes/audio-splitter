import os
import sys
import tempfile
import zipfile
import warnings
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr

# Carrega variáveis de ambiente a partir de um arquivo .env, se disponível.
try:
    from dotenv import load_dotenv
    load_dotenv()  # Procura automaticamente por um arquivo .env na raiz do projeto.
except ImportError:
    warnings.warn("python-dotenv não está instalado. Configure as variáveis de ambiente manualmente.")

# Obtenha o token do Hugging Face a partir de uma variável de ambiente.
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Variável de ambiente HF_TOKEN não configurada. Configure o token via .env ou no ambiente.")

# Configure o token e outras variáveis de ambiente necessárias.
os.environ["HF_TOKEN"] = token
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Detecta se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilizando dispositivo: {device}")

# Variável global para armazenar o pipeline de diarização (para otimizar o carregamento)
global_diarization_pipeline = None

# =============================================================================
# Função auxiliar: GPU Split on Silence
# Recria a funcionalidade de "split_on_silence" utilizando torchaudio e torch,
# de modo que os cálculos sejam feitos na GPU.
# =============================================================================
def gpu_split_on_silence(waveform, sample_rate, min_silence_len=1000, silence_thresh=-40, keep_silence=500):
    """
    Parâmetros:
      - waveform: Tensor com shape (channels, samples)
      - sample_rate: taxa de amostragem do áudio
      - min_silence_len: duração mínima do silêncio para ser considerado (ms)
      - silence_thresh: limiar em dBFS para definir silêncio (ex: -40)
      - keep_silence: quantidade (ms) de silêncio para manter nas bordas
    Retorna:
      - Uma lista de tuplas (início, fim) em samples indicando os intervalos não-silenciosos.
    """
    # Converte o limiar de dBFS para amplitude (assume sinal normalizado entre -1 e 1)
    amp_thresh = 10 ** (silence_thresh / 20.0)
    # Calcula o envelope (média do valor absoluto entre canais)
    envelope = torch.mean(torch.abs(waveform), dim=0)  # shape: (samples,)
    
    # Suaviza o envelope usando uma média móvel (kernel de 10ms)
    kernel_size = max(1, int(sample_rate * 0.01))  # 10ms em número de amostras
    envelope = envelope.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, samples)
    envelope = F.avg_pool1d(envelope, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    envelope = envelope.squeeze()  # volta para shape: (samples,)
    
    # Identifica os índices onde o áudio é considerado não-silencioso
    non_silent_idx = (envelope >= amp_thresh).nonzero(as_tuple=False).squeeze().cpu().numpy()
    if non_silent_idx.size == 0:
        return []  # nenhum som detectado
    
    # Agrupa os índices contíguos em blocos
    groups = []
    current_group = [non_silent_idx[0]]
    for idx in non_silent_idx[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    
    segments = []
    # Adiciona um "buffer" de silêncio (keep_silence) em cada borda
    pad = int(keep_silence * sample_rate / 1000)
    for group in groups:
        start = max(0, group[0] - pad)
        end = min(waveform.shape[1], group[-1] + pad)
        segments.append((start, end))
    
    # Filtra segmentos que sejam menores que o mínimo de duração (min_segment_duration será aplicado depois)
    return segments

# =============================================================================
# Função de segmentação de áudio otimizada para GPU
# =============================================================================
def process_audio_file(audio_file_path, min_segment_duration=3, silence_thresh=-40,
                       noise_threshold=-20, output_format="wav", keep_silence=500):
    """
    Realiza a segmentação do áudio utilizando operações em GPU.
    - Carrega o áudio com torchaudio.
    - Processa o envelope de áudio para detectar os intervalos não-silenciosos.
    - Separa os segmentos com base nos parâmetros informados.
    - Exporta cada segmento para um arquivo temporário.
    """
    # Carrega o áudio e move para o dispositivo (GPU se disponível)
    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)
    
    # (Opcional) Compressão de faixa dinâmica poderia ser implementada aqui.
    # Para manter similar à pydub, essa etapa foi omitida ou pode ser adicionada com transformações customizadas.
    
    # Identifica os intervalos não-silenciosos utilizando a função GPU
    segments_bounds = gpu_split_on_silence(waveform, sample_rate, min_silence_len=1000, 
                                           silence_thresh=silence_thresh, keep_silence=keep_silence)
    
    final_segments = []
    # Converte duração mínima para número de amostras
    min_samples = int(min_segment_duration * sample_rate)
    for (start, end) in segments_bounds:
        if (end - start) < min_samples:
            continue
        # Aqui poderíamos aplicar o teste de "noise_threshold" similar ao pydub.
        # Para simplicidade, mantemos o segmento se já passou no teste de silêncio.
        final_segments.append((start, end))
    
    temp_files = []
    # Exporta cada segmento para um arquivo usando torchaudio (a escrita ocorrerá na CPU)
    for k, (start, end) in enumerate(final_segments):
        segment_tensor = waveform[:, start:end].cpu()  # move para CPU para salvar
        segment_file_name = f"segment_{k+1}.{output_format}"
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        temp_file.close()
        # Salva o segmento; torchaudio.save espera que o tensor esteja no formato (channels, samples)
        torchaudio.save(temp_file.name, segment_tensor, sample_rate)
        temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
        counter = 1
        while os.path.exists(temp_file_name):
            segment_file_name = f"segment_{k+1}_{counter}.{output_format}"
            temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
            counter += 1
        os.rename(temp_file.name, temp_file_name)
        temp_files.append((segment_file_name, temp_file_name))
    
    return temp_files

# =============================================================================
# Função experimental de diarização utilizando pyannote.audio com suporte à GPU
# =============================================================================
def process_audio_file_diarization(audio_file_path, min_segment_duration=3, output_format="wav"):
    global global_diarization_pipeline
    try:
        from pyannote.audio import Pipeline
        # Inicializa o pipeline e força o uso de GPU se disponível
        if global_diarization_pipeline is None:
            global_diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization", 
                use_auth_token=os.environ["HF_TOKEN"],
                device=device
            )
    except Exception as e:
        print("Falha ao carregar pyannote.audio, utilizando método tradicional:", e)
        return process_audio_file(audio_file_path, min_segment_duration, silence_thresh=-40,
                                  noise_threshold=-20, output_format=output_format)
    
    diarization = global_diarization_pipeline(audio_file_path)
    # Carrega o áudio usando torchaudio para extrair os segmentos (pode ser otimizado com GPU, se necessário)
    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)
    segments_by_speaker = {}
    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        duration = segment.end - segment.start
        if duration < min_segment_duration:
            continue
        start_ms = int(segment.start * sample_rate)
        end_ms = int(segment.end * sample_rate)
        seg_audio = waveform[:, start_ms:end_ms].cpu()  # move para CPU para salvar
        segment_file_name = f"speaker_{speaker}_segment_{i+1}.{output_format}"
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        temp_file.close()
        torchaudio.save(temp_file.name, seg_audio, sample_rate)
        temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
        counter = 1
        while os.path.exists(temp_file_name):
            segment_file_name = f"speaker_{speaker}_segment_{i+1}_{counter}.{output_format}"
            temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
            counter += 1
        os.rename(temp_file.name, temp_file_name)
        segments_by_speaker.setdefault(speaker, []).append((segment_file_name, temp_file_name))
    return segments_by_speaker

# =============================================================================
# Função principal de processamento.
# Organiza os segmentos em pastas e constrói um dicionário para preview.
# =============================================================================
def process_audio_files(audio_files, min_segment_duration, silence_thresh, noise_threshold,
                        output_format, experimental_diarization, keep_silence):
    output_dir = tempfile.mkdtemp(prefix="audio_split_")
    preview_info = {}  # chave: nome base do arquivo; valor: se diarização -> dict {speaker: [lista de arquivos]}, senão -> lista de arquivos
    
    for file_path in audio_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        subfolder_path = os.path.join(output_dir, base_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        if experimental_diarization:
            segments_by_speaker = process_audio_file_diarization(file_path, min_segment_duration, output_format)
            if isinstance(segments_by_speaker, list):
                file_paths = []
                for seg_name, temp_path in segments_by_speaker:
                    destination = os.path.join(subfolder_path, seg_name)
                    os.rename(temp_path, destination)
                    file_paths.append(destination)
                preview_info[base_name] = file_paths
            else:
                speaker_dict = {}
                for speaker, segments in segments_by_speaker.items():
                    speaker_folder = os.path.join(subfolder_path, f"speaker_{speaker}")
                    os.makedirs(speaker_folder, exist_ok=True)
                    file_paths = []
                    for seg_name, temp_path in segments:
                        destination = os.path.join(speaker_folder, seg_name)
                        os.rename(temp_path, destination)
                        file_paths.append(destination)
                    speaker_dict[speaker] = file_paths
                preview_info[base_name] = speaker_dict
        else:
            segments = process_audio_file(file_path, min_segment_duration, silence_thresh,
                                          noise_threshold, output_format, keep_silence)
            file_paths = []
            for seg_name, temp_path in segments:
                destination = os.path.join(subfolder_path, seg_name)
                os.rename(temp_path, destination)
                file_paths.append(destination)
            preview_info[base_name] = file_paths

    zip_path = os.path.join(output_dir, "split_audio.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file == "split_audio.zip":
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname=arcname)
    
    return zip_path, preview_info, output_format

# =============================================================================
# Função para converter um arquivo em Data URI (base64)
# =============================================================================
def file_to_data_uri(file_path, out_format):
    mime_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg"
    }.get(out_format, "audio/wav")
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# =============================================================================
# Função para construir o preview em HTML (utilizando recurso do Gradio)
# =============================================================================
def build_preview_html(preview_info, experimental_diarization, out_format):
    html = ""
    for base_name, info in preview_info.items():
        html += f"<h4>{base_name}</h4>"
        if experimental_diarization:
            # info é um dicionário: chave = speaker, valor = lista de arquivos
            for speaker, file_list in info.items():
                html += f"<h5>Speaker: {speaker}</h5>"
                for file in file_list:
                    data_uri = file_to_data_uri(file, out_format)
                    html += f'<audio controls src="{data_uri}"></audio><br>'
        else:
            # info é uma lista de arquivos
            for file in info:
                data_uri = file_to_data_uri(file, out_format)
                html += f'<audio controls src="{data_uri}"></audio><br>'
    return html

# =============================================================================
# Construção da interface Gradio
# =============================================================================
with gr.Blocks() as demo:
    gr.Markdown("## Audio Splitter")
    
    with gr.Row():
        audio_input = gr.File(label="Arquivos de Áudio", file_count="multiple", type="filepath")
        min_seg_input = gr.Number(label="Duração Mínima dos Segmentos (segundos)", value=3)
    with gr.Row():
        silence_thresh_input = gr.Slider(minimum=-100, maximum=0, step=1, label="Sensibilidade para Cortar (dB)", value=-40)
        noise_threshold_input = gr.Slider(minimum=-50, maximum=0, step=1, label="Quantidade de Ruído Permitido (dB)", value=-20)
    with gr.Row():
        format_input = gr.Dropdown(label="Formato de Saída", choices=["wav", "mp3", "flac", "ogg"], value="wav")
        experimental_checkbox = gr.Checkbox(label="Experimental: Diarização de Voz", value=False)
    with gr.Row():
        keep_silence_input = gr.Slider(minimum=0, maximum=1000, step=50, label="Intervalo do Corte (ms)", value=500)
    
    process_btn = gr.Button("Processar Áudios")
    with gr.Row():
        zip_output = gr.File(label="Download do ZIP")
    gr.Markdown("### Preview dos Áudios Cortados")
    preview_output = gr.HTML(value="")
    
    def run_all(audio_files, min_seg, silence_thresh, noise_thresh, out_format, experimental, keep_silence):
        if not audio_files:
            return None, ""
        zip_path, preview_info, out_format = process_audio_files(
            audio_files, min_seg, silence_thresh, noise_thresh, out_format, experimental, keep_silence
        )
        preview_html = build_preview_html(preview_info, experimental, out_format)
        return zip_path, preview_html

    process_btn.click(
        fn=run_all,
        inputs=[audio_input, min_seg_input, silence_thresh_input, noise_threshold_input, format_input, experimental_checkbox, keep_silence_input],
        outputs=[zip_output, preview_output]
    )

if __name__ == "__main__":
    demo.launch()
