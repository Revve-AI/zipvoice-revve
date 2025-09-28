# import requests

# url = "http://localhost:8087/synthesize"
# data = {
#     "text": "Dạ, em là nhân viên chăm sóc khách hàng của, vê, e tê xê, đơn vị thu phí tự động không dừng cho xe Toyota Vios biển số tám tám H một hai ba bốn năm của anh. Dịp này, vê, e tê xê đang triển khai chương trình tri ân đặc biệt cho một nghìn khách hàng đầu tiên bên em gọi ra. Xe của anh sẽ được giảm ưu đãi sâu về phí khi tham gia bảo hiểm trách nhiệm dân sự bắt buộc. Em xin phép được báo giá nhanh và hỗ trợ bảo hiểm cho xe của mình lên ứng dụng vê e, tê xê luôn được không ạ?",
#     "speed": 1.0,
#     "playback_speed": 1.1
# }

# response = requests.post(url,  data=data)

# if response.status_code == 200:
#     # Lấy wav bytes
#     wav_bytes = response.content
#     # Lấy metadata từ headers
#     duration = response.headers.get('X-Duration')
#     message = response.headers.get('X-Message') 
#     sample_rate = response.headers.get('X-Sample-Rate')
    
#     print(f"Duration: {duration}s")
#     print(f"Message: {message}")
#     print(f"Sample Rate: {sample_rate}")
    
#     # Lưu file
#     with open("output_test.wav", "wb") as f:
#         f.write(wav_bytes)
# else:
#     print(f"Error: {response.json()}")




import requests
import numpy as np
import soundfile as sf

url = "http://localhost:8087/synthesize"
data = {
    "text": "Dạ, em là nhân viên chăm sóc khách hàng của, VETC đơn vị thu phí tự động không dừng cho xe Toyota Vios biển số tám ",
    "speed": 1.0,
    "playback_speed": 1.1
}

response = requests.post(url, data=data)

if response.status_code == 200:
    result = response.json()
    
    # Chuyển list thành numpy array
    audio_numpy = np.array(result["audio"], dtype=np.float32)
    sample_rate = result["sample_rate"]
    
    # Lưu file WAV
    sf.write("output_test.wav", audio_numpy, sample_rate)
    
    print(f"✓ Saved: output_test.wav")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Sample Rate: {sample_rate} Hz")
else:
    print(f"Error: {response.json()}")