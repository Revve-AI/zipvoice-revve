



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
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill

from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from replace_with_mapping import normalize_text,split_text_into_chunks
from infer import load_model,generate_speech,generate_speech_chunked,apply_ffmpeg_speed
model = None
vocoder = None
tokenizer = None
feature_extractor = None
device = "cuda"
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

# FastAPI app
app = FastAPI(title="ZipVoice TTS API", description="Text-to-Speech API using ZipVoice")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model= load_model()
# @app.post("/synthesize")
# async def synthesize_speech(
#     # prompt_text: str = Form(...),
#     text: str = Form(...),
#     speed: float = Form(1.0),
#     playback_speed: float = Form(1.0),
#     num_step: int = Form(16),
#     guidance_scale: float = Form(1.0),
#     t_shift: float = Form(0.5),
#     chunk_size: int = Form(30),
#     # prompt_wav: UploadFile = File(...)
# ):
#     """
#     Synthesize speech from text with voice cloning
    
#     Returns wav audio bytes directly
    
#     - **prompt_text**: Text tương ứng với file audio mẫu
#     - **prompt_wav**: File audio mẫu (.wav)
#     - **text**: Text cần synthesize
#     - **speed**: Tốc độ nói (0.5-2.0)
#     - **playback_speed**: Tốc độ phát lại (0.5-2.0)
#     - **num_step**: Số bước sampling (8-64)
#     - **guidance_scale**: Guidance scale (0.5-5.0)
#     - **t_shift**: Time shift (0.1-1.0)
#     - **chunk_size**: Số từ mỗi đoạn khi chia text dài (10-50)
#     """
#     prompt_text="tao vừa gặp lại bạn cũ hồi cấp ba, lâu rồi không gặp mà vẫn nói chuyện hợp như ngày xưa."
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model chưa được load")
    
#     if not text.strip():
#         raise HTTPException(status_code=400, detail="Text không được rỗng")
    

    
#     temp_prompt_path = None
#     temp_output_path = None
#     temp_final_path = None
#     print("text : ",text)

#     text=normalize_text(text)
#     print("text chuan hoa :",text)
#     try:
#         # Lưu file prompt wav tạm thời
#         temp_prompt_path="/opt/shared_ai_data/sythesized_data_gemini_tts_madebytantran/dataset_pro_1/audio/Aoede/everyday_chatgpt/00070.wav"
#     #     (
#     #     "tao vừa gặp lại bạn cũ hồi cấp ba, lâu rồi không gặp mà vẫn nói chuyện hợp như ngày xưa.",
#     #     "/opt/shared_ai_data/sythesized_data_gemini_tts_madebytantran/dataset_pro_1/audio/Aoede/everyday_chatgpt/00070.wav",
#     #     0.9
#     # ),
#     # (
#     #     "nói chung thì việc đầu tư vào giáo dục luôn mang lại lợi ích lâu dài cho xã hội",
#     #     "/opt/shared_ai_data/sythesized_data_gemini_tts_madebytantran/dataset_pro_1/audio/Aoede/everyday_claude/00145.wav",
#     #     0.9
#     # ),
        
#         # Đếm số từ trong text
#         word_count = len(text.split())
#         chunk_size=30
#         if word_count <= chunk_size:
#             print("Text ngắn")
#             # Text ngắn, generate trực tiếp
#             wav, sr = generate_speech(
#                 prompt_text=prompt_text,
#                 prompt_wav_path=temp_prompt_path,
#                 text=text,
#                 speed=speed,
#                 num_step=num_step,
#                 guidance_scale=guidance_scale,
#                 t_shift=t_shift,
#             )
            
#             if wav is None:
#                 raise HTTPException(status_code=500, detail="Không thể generate audio")
            
        
#         else:
#             # Text dài, sử dụng chunking
#             print("Text dài")
#             wav, sr= generate_speech_chunked(
#                 prompt_text=prompt_text,
#                 prompt_wav_path=temp_prompt_path,
#                 text=text,
#                 speed=speed,
#                 num_step=num_step,
#                 guidance_scale=guidance_scale,
#                 t_shift=t_shift,
#                 chunk_size=chunk_size,
#             )
            
#             if wav is None:
#                 raise HTTPException(status_code=500)
        
#         # Lưu file tạm thời để xử lý
#         temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#         if wav.ndim == 1:
#             wav_to_save = wav.unsqueeze(0)
#         else:
#             wav_to_save = wav
#         torchaudio.save(temp_output_path, wav_to_save, sample_rate=sr)
        
#         # Áp dụng playback speed nếu cần
#         temp_final_path = temp_output_path
#         if playback_speed is not None and abs(playback_speed - 1.0) > 1e-8:
#             temp_final_path = apply_ffmpeg_speed(temp_output_path, playback_speed)
        
#         # Đọc file wav thành bytes
#         with open(temp_final_path, "rb") as f:
#             wav_bytes = f.read()
#         print(wav_bytes)
#         # Tính thời lượng
#         duration = wav.shape[1] / sr if wav.ndim > 1 else len(wav) / sr
        
#         # Trả về wav bytes với headers phù hợp
   
#         return Response(
#             content=wav_bytes,
#             media_type="audio/wav",
#             headers={
#                 "Content-Disposition": "attachment; filename=synthesized_audio.wav",
#                 "X-Duration": str(duration),
#                 "X-Sample-Rate": str(sr)
#             }
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error in synthesize_speech: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")
    
#     finally:
#         # Cleanup tất cả file tạm
#         for temp_file in [ temp_output_path, temp_final_path]:
#             try:
#                 if temp_file and os.path.exists(temp_file) and temp_file != temp_output_path:
#                     os.remove(temp_file)
#             except Exception as e:
#                 print(f"Warning: Could not remove temp file {temp_file}: {e}")
        
#         # Cleanup riêng cho temp_output_path
#         try:
#             if temp_output_path and os.path.exists(temp_output_path):
#                 os.remove(temp_output_path)
#         except Exception as e:
#             print(f"Warning: Could not remove temp output file: {e}")

@app.post("/synthesize")
async def synthesize_speech(
    # prompt_text: str = Form(...),
    text: str = Form(...),
    speed: float = Form(1.0),
    playback_speed: float = Form(1.0),
    num_step: int = Form(16),
    guidance_scale: float = Form(1.0),
    t_shift: float = Form(0.5),
    chunk_size: int = Form(30),
    # prompt_wav: UploadFile = File(...)
):
    """
    Synthesize speech from text with voice cloning
    
    Returns audio as numpy array and sample rate in JSON format
    
    - **prompt_text**: Text tương ứng với file audio mẫu
    - **prompt_wav**: File audio mẫu (.wav)
    - **text**: Text cần synthesize
    - **speed**: Tốc độ nói (0.5-2.0)
    - **playback_speed**: Tốc độ phát lại (0.5-2.0)
    - **num_step**: Số bước sampling (8-64)
    - **guidance_scale**: Guidance scale (0.5-5.0)
    - **t_shift**: Time shift (0.1-1.0)
    - **chunk_size**: Số từ mỗi đoạn khi chia text dài (10-50)
    """
    import numpy as np
    import tempfile
    import subprocess
    
    prompt_text = "tao vừa gặp lại bạn cũ hồi cấp ba, lâu rồi không gặp mà vẫn nói chuyện hợp như ngày xưa."
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    
    temp_prompt_path = None
    temp_output_path = None
    temp_final_path = None
    print("text : ", text)

    text = normalize_text(text)
    print("text chuan hoa :", text)
    
    try:
        # Lưu file prompt wav tạm thời
        temp_prompt_path = "/opt/shared_ai_data/sythesized_data_gemini_tts_madebytantran/dataset_pro_1/audio/Aoede/everyday_chatgpt/00070.wav"
        
        # Đếm số từ trong text
        word_count = len(text.split())
        chunk_size = 30
        
        if word_count <= chunk_size:
            print("Text ngắn")
            # Text ngắn, generate trực tiếp
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
                raise HTTPException(status_code=500, detail="Không thể generate audio")
        
        else:
            # Text dài, sử dụng chunking
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
                raise HTTPException(status_code=500)
        
        # Chuyển đổi từ tensor sang numpy array
        if hasattr(wav, 'numpy'):
            # Nếu wav là tensor PyTorch
            audio_numpy = wav.detach().cpu().numpy()
        else:
            # Nếu wav đã là numpy array
            audio_numpy = wav
        
        # Đảm bảo audio là 1D array
        if audio_numpy.ndim > 1:
            # Nếu có nhiều channel, lấy channel đầu tiên hoặc trung bình
            if audio_numpy.shape[0] == 1:
                audio_numpy = audio_numpy[0]  # Lấy channel đầu tiên
            else:
                audio_numpy = np.mean(audio_numpy, axis=0)  # Trung bình các channel
        
        # Áp dụng playback speed nếu cần
        if playback_speed is not None and abs(playback_speed - 1.0) > 1e-8:
            # Tạo file tạm để xử lý speed với ffmpeg
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Lưu audio tạm thời
            if hasattr(wav, 'unsqueeze'):
                wav_to_save = wav.unsqueeze(0) if wav.ndim == 1 else wav
            else:
                wav_to_save = torch.from_numpy(audio_numpy).unsqueeze(0)
            
            torchaudio.save(temp_output_path, wav_to_save, sample_rate=sr)
            
            # Áp dụng speed change
            temp_final_path = apply_ffmpeg_speed(temp_output_path, playback_speed)
            
            # Đọc lại audio sau khi thay đổi speed
            wav_speed_changed, sr_speed = torchaudio.load(temp_final_path)
            audio_numpy = wav_speed_changed.squeeze().numpy()
            sr = sr_speed
        
        # Tính thời lượng
        duration = len(audio_numpy) / sr
        
        print(f"Audio shape: {audio_numpy.shape}, Sample rate: {sr}, Duration: {duration}s")
        
        # Trả về JSON chứa audio array và metadata
        print(audio_numpy)
        return {
            "audio": audio_numpy.tolist(),  # Chuyển numpy array thành list để serialize JSON
            "sample_rate": int(sr),
            "duration": float(duration),
            "shape": list(audio_numpy.shape),
            "dtype": str(audio_numpy.dtype),
            "min_value": float(np.min(audio_numpy)),
            "max_value": float(np.max(audio_numpy))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in synthesize_speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")
    
    finally:
        # Cleanup tất cả file tạm
        for temp_file in [temp_output_path, temp_final_path]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8087)




