import os
import sys
import tempfile
import zipfile
import warnings
import base64
from pydub import AudioSegment, silence, effects
import gradio as gr

# O token que está sendo utilizado é de uso pessoal, e deve ser substituído caso o projeto vire público, além de não estar hard-coded.
os.environ["HF_TOKEN"] = "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"

# Suprime alguns warnings que não interferem na execução
warnings.filterwarnings("ignore", message="Using SYMLINK strategy on Windows")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Requested Pretrainer collection using symlinks")

# Para evitar problemas com symlink no Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Variável global para armazenar o pipeline de diarização (para otimizar o carregamento)
global_diarization_pipeline = None


# Função tradicional de segmentação (separa com base em períodos de silêncio)
def process_audio_file(audio_file_path, min_segment_duration=3, silence_thresh=-40,
                       noise_threshold=-20, output_format="wav", keep_silence=500):
    
    audio = AudioSegment.from_file(audio_file_path)
    audio = effects.compress_dynamic_range(audio)
    
    segments = silence.split_on_silence(
        audio, 
        min_silence_len=1000, 
        silence_thresh=silence_thresh, 
        keep_silence=keep_silence
    )
    final_segments = []
    for seg in segments:
        if len(seg) < min_segment_duration * 1000:
            continue
        if seg.dBFS is not None and seg.dBFS > noise_threshold:
            sub_segments = silence.split_on_silence(
                seg, 
                min_silence_len=300, 
                silence_thresh=silence_thresh, 
                keep_silence=keep_silence
            )
            for sub_seg in sub_segments:
                if len(sub_seg) >= min_segment_duration * 1000:
                    final_segments.append(sub_seg)
        else:
            final_segments.append(seg)
    
    temp_files = []
    for k, segment in enumerate(final_segments):
        segment_file_name = f"segment_{k+1}.{output_format}"
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        segment.export(temp_file.name, format=output_format)
        temp_file.close()
        temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
        counter = 1
        while os.path.exists(temp_file_name):
            segment_file_name = f"segment_{k+1}_{counter}.{output_format}"
            temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
            counter += 1
        os.rename(temp_file.name, temp_file_name)
        temp_files.append((segment_file_name, temp_file_name))
    
    return temp_files


# Função experimental de diarização utilizando pyannote.audio
def process_audio_file_diarization(audio_file_path, min_segment_duration=3, output_format="wav"):
    global global_diarization_pipeline
    try:
        from pyannote.audio import Pipeline
        token = os.environ.get("HF_TOKEN") 
        if token is None:
            raise ValueError("Variável de ambiente HF_TOKEN não configurada.") 
        if global_diarization_pipeline is None:
            global_diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    except Exception as e:
        print("Falha ao carregar pyannote.audio, utilizando método tradicional:", e)
        return process_audio_file(audio_file_path, min_segment_duration, silence_thresh=-40,
                                  noise_threshold=-20, output_format=output_format)
    
    diarization = global_diarization_pipeline(audio_file_path)
    audio = AudioSegment.from_file(audio_file_path)
    segments_by_speaker = {}
    for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        duration = segment.end - segment.start
        if duration < min_segment_duration:
            continue
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        seg_audio = audio[start_ms:end_ms]
        segment_file_name = f"speaker_{speaker}_segment_{i+1}.{output_format}"
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        seg_audio.export(temp_file.name, format=output_format)
        temp_file.close()
        temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
        counter = 1
        while os.path.exists(temp_file_name):
            segment_file_name = f"speaker_{speaker}_segment_{i+1}_{counter}.{output_format}"
            temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
            counter += 1
        os.rename(temp_file.name, temp_file_name)
        segments_by_speaker.setdefault(speaker, []).append((segment_file_name, temp_file_name))
    return segments_by_speaker

# Função principal de processamento. Organiza os segmentos em pastas e constrói um dicionário para preview
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

# Função para converter um arquivo em Data URI (base64)
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


# Função para construir o preview em HTML (utilizando o recurso do próprio gradio não fica muito bom.)
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

# Construção do Gradio
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