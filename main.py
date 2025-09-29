import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, List
import time
import torch
import torchaudio
import safetensors.torch
from vocos import Vocos
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill

from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from replace_with_mapping import normalize_text,split_text_into_chunks
from infer import load_model,generate_speech,generate_speech_chunked,apply_ffmpeg_speed
import numpy as np

# Global variables
model = None
vocoder = None
tokenizer = None
feature_extractor = None
device = "cuda"
model_config = None

def init_model():
    """Initialize model"""
    global model
    if model is None:
        model = load_model()
        print("Model loaded successfully")

def synthesize_speech(
    text: str,
    speed: float = 1.0,
    playback_speed: float = 1.0,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    chunk_size: int = 30,
):
    """
    Synthesize speech from text with voice cloning
    
    Returns: (audio_numpy_array, sample_rate, metadata)
    
    Args:
        text: Text cần synthesize
        speed: Tốc độ nói (0.5-2.0)
        playback_speed: Tốc độ phát lại (0.5-2.0)
        num_step: Số bước sampling (8-64)
        guidance_scale: Guidance scale (0.5-5.0)
        t_shift: Time shift (0.1-1.0)
        chunk_size: Số từ mỗi đoạn khi chia text dài (10-50)
    """
    
    prompt_text = "tao vừa gặp lại bạn cũ hồi cấp ba, lâu rồi không gặp mà vẫn nói chuyện hợp như ngày xưa."
    
    if model is None:
        raise Exception("Model chưa được load. Gọi init_model() trước.")
    
    if not text.strip():
        raise Exception("Text không được rỗng")
    
    temp_output_path = None
    temp_final_path = None
    print("text :", text)

    text = normalize_text(text)
    print("text chuan hoa :", text)
    
    try:
        temp_prompt_path = "00070.wav"
        
        # Đếm số từ trong text
        word_count = len(text.split())
        
        if word_count <= chunk_size:
            print("Text ngắn")
            wav, sr = generate_speech(
                prompt_text=prompt_text,
                prompt_wav_path=temp_prompt_path,
                text=text,
                speed=speed,
                num_step=num_step,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
            )
            
            if wav is None:
                raise Exception("Không thể generate audio")
        
        else:
            print("Text dài")
            wav, sr = generate_speech_chunked(
                prompt_text=prompt_text,
                prompt_wav_path=temp_prompt_path,
                text=text,
                speed=speed,
                num_step=num_step,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                chunk_size=chunk_size,
            )
            
            if wav is None:
                raise Exception("Không thể generate audio")
        
    
        if hasattr(wav, 'numpy'):
            audio_numpy = wav.detach().cpu().numpy()
        else:
            audio_numpy = wav
        
      
        if audio_numpy.ndim > 1:
            if audio_numpy.shape[0] == 1:
                audio_numpy = audio_numpy[0]
            else:
                audio_numpy = np.mean(audio_numpy, axis=0)
        
        # Áp dụng playback speed nếu cần
        if playback_speed is not None and abs(playback_speed - 1.0) > 1e-8:
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            if hasattr(wav, 'unsqueeze'):
                wav_to_save = wav.unsqueeze(0) if wav.ndim == 1 else wav
            else:
                wav_to_save = torch.from_numpy(audio_numpy).unsqueeze(0)
            
            torchaudio.save(temp_output_path, wav_to_save, sample_rate=sr)
            temp_final_path = apply_ffmpeg_speed(temp_output_path, playback_speed)
            
            wav_speed_changed, sr_speed = torchaudio.load(temp_final_path)
            audio_numpy = wav_speed_changed.squeeze().numpy()
            sr = sr_speed
        
        # Tính thời lượng
        duration = len(audio_numpy) / sr
        
        print(f"Audio shape: {audio_numpy.shape}, Sample rate: {sr}, Duration: {duration}s")
        
        # Metadata
        metadata = {
            "sample_rate": int(sr),
            "duration": float(duration),
            "shape": list(audio_numpy.shape),
            "dtype": str(audio_numpy.dtype),
            "min_value": float(np.min(audio_numpy)),
            "max_value": float(np.max(audio_numpy))
        }
        
        return audio_numpy, sr, metadata
        
    except Exception as e:
        print(f"Error in synthesize_speech: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # Cleanup
        for temp_file in [temp_output_path, temp_final_path]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {e}")

def save_audio(audio_numpy, sample_rate, filename="output.wav"):
    """Lưu audio thành file WAV"""
    import soundfile as sf
    sf.write(filename, audio_numpy, sample_rate)
    print(f" Saved: {filename}")
if __name__ == "__main__":
    # Khởi tạo model
    init_model()
    
    text = "Dạ, em là nhân viên chăm sóc khách hàng của, vê e tê xê, đơn vị thu phí tự động không dừng cho xe Toyota Vios biển số tám tám H một hai ba bốn năm của anh."
    
    try:
        # Synthesize
        audio_numpy, sample_rate, metadata = synthesize_speech(
            text=text,
            speed=1.0,
            playback_speed=1.1
        )
        print("audio_numpy",audio_numpy)
        save_audio(audio_numpy, sample_rate, "output.wav")
        
    except Exception as e:
        print(f"Error: {e}")