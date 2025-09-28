
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
from pydantic import BaseModel
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
from lhotse.utils import fix_random_seed
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
import subprocess
import shutil
import re

from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from replace_with_mapping import normalize_text,split_text_into_chunks
# Cấu hình model của bạn
CONFIG = {
    "model_name": "zipvoice",  # hoặc "zipvoice_distill"
    "model_dir": "model/",  # Thư mục model của bạn
    "checkpoint_name": "/home/cuong/TTS/a/ZipVoice/model/checkpoint-50000.pt",  # Tên checkpoint
    "tokenizer": "espeak",  # emilia, libritts, espeak, simple
    "lang": "vi",  # Chỉ dùng cho espeak tokenizer
    "vocoder_path": None,  # None để dùng vocoder mặc định từ HuggingFace
    "feat_scale": 0.1,
    "target_rms": 0.1,
    "sampling_rate": 24000,
    "seed": 666,
    "chunk_size": 30,  # Số từ mỗi đoạn
}
# Biến global để lưu model và các thành phần
model = None
vocoder = None
tokenizer = None
feature_extractor = None
device = None
model_config = None

# Pydantic models for request/response
class SynthesizeRequest(BaseModel):
    prompt_text: str
    text: str
    speed: float = 1.0
    playback_speed: float = 1.0
    num_step: int = 16
    guidance_scale: float = 1.0
    t_shift: float = 0.5
    chunk_size: int = 30

class SynthesizeResponse(BaseModel):
    success: bool
    message: str
    duration: float = None
def apply_ffmpeg_speed(input_path: str, speed_factor: float, timeout: int = 60) -> str:
    """
    Sử dụng FFmpeg để thay đổi tốc độ audio mà không thay đổi pitch.
    """
    try:
        if abs(speed_factor - 1.0) < 1e-8:
            return input_path

        if shutil.which("ffmpeg") is None:
            print("⚠️ FFmpeg không được tìm thấy trên PATH. Bỏ qua speed adjustment.")
            return input_path

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-filter:a", f"atempo={speed_factor}",
            "-ar", str(CONFIG.get("sampling_rate", 24000)),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print(f"FFmpeg speed adjustment thành công: {speed_factor}x -> {output_path}")
            return output_path
        else:
            print("FFmpeg returned error:", result.stderr or result.stdout)
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return input_path

    except subprocess.TimeoutExpired:
        print("FFmpeg timeout")
        return input_path
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return input_path

import re
from difflib import SequenceMatcher
def get_vocoder(vocos_local_path: Optional[str] = None):
    """Load vocoder"""
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder
def load_model():
    """Load model một lần khi khởi tạo"""
    global model, vocoder, tokenizer, feature_extractor, device, model_config
    
    try:
        print("🔄 Đang load ZipVoice model...")
        
        fix_random_seed(CONFIG["seed"])
        
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"📱 Sử dụng device: {device}")
        
        model_dir = Path(CONFIG["model_dir"])
        model_ckpt = model_dir / CONFIG["checkpoint_name"]
        model_config_file = model_dir / "model.json"
        token_file = model_dir / "tokens.txt"
        
        print(f"📁 Sử dụng model local: {model_dir}")
        
        for filename, filepath in [
            ("checkpoint", model_ckpt),
            ("config", model_config_file), 
            ("tokens", token_file)
        ]:
            if not filepath.exists():
                print(f"❌ Không tìm thấy {filename}: {filepath}")
                return False
        
        print("🔤 Load tokenizer...")
        if CONFIG["tokenizer"] == "emilia":
            tokenizer = EmiliaTokenizer(token_file=token_file)
        elif CONFIG["tokenizer"] == "libritts":
            tokenizer = LibriTTSTokenizer(token_file=token_file)
        elif CONFIG["tokenizer"] == "espeak":
            tokenizer = EspeakTokenizer(token_file=token_file, lang=CONFIG["lang"])
        else:
            tokenizer = SimpleTokenizer(token_file=token_file)
        
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size, 
            "pad_id": tokenizer.pad_id
        }
        
        with open(model_config_file, "r") as f:
            model_config = json.load(f)
        
        print("🧠 Tạo model...")
        if CONFIG["model_name"] == "zipvoice":
            model = ZipVoice(
                **model_config["model"],
                **tokenizer_config,
            )
        else:
            model = ZipVoiceDistill(
                **model_config["model"],
                **tokenizer_config,
            )
        
        print("📥 Load checkpoint...")
        if str(model_ckpt).endswith(".safetensors"):
            safetensors.torch.load_model(model, model_ckpt)
        elif str(model_ckpt).endswith(".pt"):
            load_checkpoint(filename=model_ckpt, model=model, strict=True)
        else:
            print(f"❌ Không hỗ trợ format checkpoint: {model_ckpt}")
            return False
        
        model = model.to(device)
        model.eval()
        
        print("🎵 Load vocoder...")
        vocoder = get_vocoder(CONFIG["vocoder_path"])
        vocoder = vocoder.to(device)
        vocoder.eval()
        
        print("🎛️ Load feature extractor...")
        if model_config["feature"]["type"] == "vocos":
            feature_extractor = VocosFbank()
        else:
            print(f"❌ Không hỗ trợ feature type: {model_config['feature']['type']}")
            return False
        
        CONFIG["sampling_rate"] = model_config["feature"]["sampling_rate"]
        
        print("✅ Model đã được load thành công!")
        return model
        
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_speech(
    prompt_text: str,
    prompt_wav_path: str,
    text: str,
    speed: float = 1.0,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
):
    """Generate speech từ text sử dụng model đã load"""
    global model, vocoder, tokenizer, feature_extractor, device
    
    try:
        tokens = tokenizer.texts_to_token_ids([text])
        prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
        
        prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav_path)
        
        if prompt_sampling_rate != CONFIG["sampling_rate"]:
            resampler = torchaudio.transforms.Resample(
                orig_freq=prompt_sampling_rate, 
                new_freq=CONFIG["sampling_rate"]
            )
            prompt_wav = resampler(prompt_wav)
        
        prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
        if prompt_rms < CONFIG["target_rms"]:
            prompt_wav = prompt_wav * CONFIG["target_rms"] / prompt_rms
        
        prompt_features = feature_extractor.extract(
            prompt_wav, sampling_rate=CONFIG["sampling_rate"]
        ).to(device)
        
        prompt_features = prompt_features.unsqueeze(0) * CONFIG["feat_scale"]
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
        
        with torch.inference_mode():
            (
                pred_features,
                pred_features_lens,
                pred_prompt_features,
                pred_prompt_features_lens,
            ) = model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
                t_shift=t_shift,
                duration="predict",
                num_step=num_step,
                guidance_scale=guidance_scale,
            )
            
            pred_features = pred_features.permute(0, 2, 1) / CONFIG["feat_scale"]
            wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
            
            if prompt_rms < CONFIG["target_rms"]:
                wav = wav * prompt_rms / CONFIG["target_rms"]
        
        return wav.cpu(), CONFIG["sampling_rate"]
        
    except Exception as e:
        print(f"❌ Lỗi khi generate speech: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_speech_chunked(
    prompt_text: str,
    prompt_wav_path: str,
    text: str,
    speed: float = 1.0,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    chunk_size: int = 20,
):
    """Generate speech từ text dài bằng cách chia thành các đoạn nhỏ"""
    try:
        all_wavs = []
    
        chunks = split_text_into_chunks(text, chunk_size)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
                print(f"Generate đoạn {i+1}/{total_chunks}: {chunk}")
                wav, sr = generate_speech(
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                    text=chunk,
                    speed=speed,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                    t_shift=t_shift,
                )
            
                all_wavs.append(wav)

       
        print("🔗 Ghép các đoạn audio...")
        silence_duration = 0.1
        silence_samples = int(silence_duration * CONFIG["sampling_rate"])
        silence = torch.zeros(1, silence_samples)
        
        final_wav_parts = []
        for i, wav in enumerate(all_wavs):
            final_wav_parts.append(wav)
            # Thêm khoảng lặng giữa các đoạn (trừ đoạn cuối)
            if i < len(all_wavs) - 1:
                final_wav_parts.append(silence)
        
        final_wav = torch.cat(final_wav_parts, dim=1)
        
        total_duration = final_wav.shape[1] / CONFIG["sampling_rate"]
        
        return final_wav, CONFIG["sampling_rate"]
     
        
    except Exception as e:
        print(f"❌ Lỗi khi generate speech chunked: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Lỗi: {e}"


