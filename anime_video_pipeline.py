# !pip install opencv-python moviepy diffusers transformers accelerate safetensors scipy --quiet
# import subprocess
# subprocess.check_call(["opencv-python", "moviepy", "diffusers", "transformers", "accelerate", "safetensors"])

import tkinter as tk
from tkinter import filedialog

def upload_file(file_type):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type} file",
        filetypes=[(f"{file_type.upper()} files", f"*.{file_type.lower()}"), ("All files", "*.*")]
    )
    return file_path

# video_path is already obtained from the upload_file function
video_path = upload_file("mp4")

print("📤 Upload nhạc nền (.mp3) hoặc bỏ qua nếu không cần")
audio_path = upload_file("mp3") if input("Do you want to upload audio? (y/n): ").lower() == 'y' else None


import cv2
import os

video_path = video_path
frame_output_dir = "frames"
os.makedirs(frame_output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  # 1 ảnh mỗi giây

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        filename = os.path.join(frame_output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"✅ Đã trích xuất {saved_count} ảnh.")



from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

style_prompt_map = {
    "waifu": "anime style beautiful girl",
    "chibi": "chibi anime character, cute, small body, big eyes",
    "cyberpunk": "cyberpunk anime style character, neon lights",
    "samurai": "samurai anime character with sword, dramatic lighting",
    "ghibli": "anime style like studio ghibli, scenic background"
}

# Chọn style ở đây:
chosen_style = "ghibli"  # <-- thay đổi tại đây nếu muốn

pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion", torch_dtype=torch.float16, revision="fp16"
).to("cuda")

anime_output_dir = "anime_frames"
os.makedirs(anime_output_dir, exist_ok=True)

for filename in sorted(os.listdir("frames")):
    img = Image.open(os.path.join("frames", filename)).convert("RGB")
    prompt = style_prompt_map[chosen_style]
    out = pipe(prompt, image=img.resize((512, 512))).images[0]
    out.save(os.path.join(anime_output_dir, filename))

print("✅ Đã xử lý ảnh Anime.")



from PIL import Image

aspect_mode = "9:16"  # hoặc "16:9"
size = (608, 1080) if aspect_mode == "9:16" else (1280, 720)

resized_dir = "resized_frames"
os.makedirs(resized_dir, exist_ok=True)

for filename in sorted(os.listdir("anime_frames")):
    img = Image.open(os.path.join("anime_frames", filename)).resize(size)
    img.save(os.path.join(resized_dir, filename))

print(f"✅ Đã resize ảnh theo tỉ lệ {aspect_mode}.")


image_files = sorted([os.path.join("resized_frames", f) for f in os.listdir("resized_frames")])
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape
video_out = "anime_output_no_audio.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

for image in image_files:
    frame = cv2.imread(image)
if audio_path:
    audio_path = audio_path
video.release()
print(f"✅ Đã ghép thành video: {video_out}")



from moviepy.editor import VideoFileClip, AudioFileClip

video = VideoFileClip(video_out)
if uploaded_audio:
    audio_path = list(uploaded_audio.keys())[0]
    audio = AudioFileClip(audio_path).subclip(0, video.duration)
    final_video = video.set_audio(audio)
else:
    final_video = video

final_video.write_videofile("final_anime_video.mp4", codec="libx264", audio_codec="aac")



from google.colab import files
files.download("final_anime_video.mp4")


